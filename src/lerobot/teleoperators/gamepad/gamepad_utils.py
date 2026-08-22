#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lerobot.utils.import_utils import _hidapi_available, _pygame_available, require_package
from lerobot.utils.keyboard_input import pynput_can_capture

from ..utils import TeleopEvents

if TYPE_CHECKING or _pygame_available:
    import pygame
else:
    pygame = None  # type: ignore[assignment]

if TYPE_CHECKING or _hidapi_available:
    import hid
else:
    hid = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Sony PlayStation controllers (DualShock 4 / DualSense / DualSense Edge).
SONY_VENDOR_ID = 0x054C
SONY_DUALSENSE_PRODUCT_IDS = frozenset({0x0CE6, 0x0DF2})
SONY_DS4_PRODUCT_IDS = frozenset({0x05C4, 0x09CC, 0x0BA0})
SONY_GAMEPAD_PRODUCT_IDS = SONY_DUALSENSE_PRODUCT_IDS | SONY_DS4_PRODUCT_IDS

# USB HID Generic Desktop usages: Joystick (0x04) and Game Pad (0x05).
HID_GENERIC_DESKTOP_PAGE = 0x01
HID_JOYSTICK_USAGE = 0x04
HID_GAMEPAD_USAGE = 0x05
HID_MOUSE_USAGE = 0x02

GAMEPAD_NAME_TOKENS = (
    "xbox",
    "x-box",
    "dualsense",
    "dualshock",
    "wireless controller",
    "ps5",
    "ps4",
    "playstation",
    "logitech",
    "gamepad",
    "controller",
)

# Analog trigger threshold after normalizing to 0..1.
TRIGGER_PRESS_THRESHOLD = 0.1


@dataclass(frozen=True)
class JoystickLayout:
    """Raw pygame.joystick axis/button indices when SDL GameController is unavailable."""

    left_x: int = 0
    left_y: int = 1
    right_x: int = 2
    right_y: int = 3
    trigger_left: int | None = 4
    trigger_right: int | None = 5
    button_south: int = 0  # A / Cross
    button_east: int = 1  # B / Circle
    button_west: int = 2  # X / Square
    button_north: int = 3  # Y / Triangle
    button_l1: int = 4
    button_r1: int = 5


# DualSense / DualShock over Windows DirectInput (pygame.joystick, not XInput).
# Face buttons are Square, Cross, Circle, Triangle — not Xbox A/B/X/Y order.
DUALSENSE_JOYSTICK_LAYOUT = JoystickLayout(
    left_x=0,
    left_y=1,
    right_x=2,
    right_y=3,
    trigger_left=4,
    trigger_right=5,
    button_west=0,
    button_south=1,
    button_east=2,
    button_north=3,
    button_l1=4,
    button_r1=5,
)

# Xbox / XInput and SDL GameController-style numbering.
XBOX_JOYSTICK_LAYOUT = JoystickLayout()


def apply_deadzone(value: float, deadzone: float) -> float:
    """Zero out small axis values and clamp the rest to [-1, 1]."""
    if abs(value) < deadzone:
        return 0.0
    return max(-1.0, min(1.0, float(value)))


def normalize_stick(raw: float, deadzone: float, *, invert_y: bool = False) -> float:
    value = -raw if invert_y else raw
    return apply_deadzone(value, deadzone)


def layout_for_device_name(name: str) -> JoystickLayout:
    """Pick a raw-joystick layout from the device product name."""
    lowered = name.lower()
    sony_tokens = (
        "dualsense",
        "dualshock",
        "wireless controller",
        "ps5",
        "ps4",
        "playstation",
        "sony",
    )
    if any(token in lowered for token in sony_tokens):
        return DUALSENSE_JOYSTICK_LAYOUT
    return XBOX_JOYSTICK_LAYOUT


def _u8_axis(value: int) -> float:
    """Normalize an unsigned 8-bit stick axis (0-255, center 128) to [-1, 1]."""
    return (value - 128) / 128.0


def _s16_axis(value: int) -> float:
    """Normalize a signed 16-bit stick axis to [-1, 1]."""
    return value / 32768.0


@dataclass
class ParsedGamepadReport:
    """Normalized pad state shared by HID parsers."""

    left_x: float = 0.0
    left_y: float = 0.0
    right_x: float = 0.0
    right_y: float = 0.0
    trigger_left: float = 0.0
    trigger_right: float = 0.0
    l1: bool = False
    r1: bool = False
    south: bool = False
    west: bool = False
    north: bool = False


def parse_xbox_gip_report(data: list[int] | bytes) -> ParsedGamepadReport | None:
    """Parse an Xbox One / Series GIP input report (18 bytes, packet type 0x20)."""
    if len(data) != 18 or data[0] != 0x20:
        return None

    raw = bytes(data)
    lx = _s16_axis(struct.unpack_from("<h", raw, 10)[0])
    ly = _s16_axis(struct.unpack_from("<h", raw, 12)[0])
    rx = _s16_axis(struct.unpack_from("<h", raw, 14)[0])
    ry = _s16_axis(struct.unpack_from("<h", raw, 16)[0])
    lt = struct.unpack_from("<H", raw, 6)[0] / 1023.0
    rt = struct.unpack_from("<H", raw, 8)[0] / 1023.0
    btn0 = data[4]
    btn1 = data[5]
    return ParsedGamepadReport(
        left_x=lx,
        # Xbox HID Y is up-positive; convert to SDL (up-negative).
        left_y=-ly,
        right_x=rx,
        right_y=-ry,
        trigger_left=lt,
        trigger_right=rt,
        l1=bool(btn1 & (1 << 4)),
        r1=bool(btn1 & (1 << 5)),
        south=bool(btn0 & (1 << 4)),  # A
        west=bool(btn0 & (1 << 6)),  # X
        north=bool(btn0 & (1 << 7)),  # Y
    )


def parse_logitech_report(data: list[int] | bytes) -> ParsedGamepadReport | None:
    """Parse a Logitech RumblePad 2-style 8-bit HID report."""
    if len(data) < 8:
        return None
    buttons = data[5]
    hats = data[6]
    return ParsedGamepadReport(
        left_x=_u8_axis(data[2]),
        left_y=_u8_axis(data[1]),
        right_x=_u8_axis(data[3]),
        right_y=_u8_axis(data[4]),
        trigger_left=1.0 if hats in (4, 6, 12) else 0.0,
        trigger_right=1.0 if hats in (8, 10, 12) else 0.0,
        l1=False,
        r1=hats in (2, 6, 10, 14),
        south=bool(buttons & (1 << 4)),
        west=bool(buttons & (1 << 5)),
        north=bool(buttons & (1 << 7)),
    )


def parse_sony_hid_report(
    data: list[int] | bytes, product_id: int | None = None
) -> ParsedGamepadReport | None:
    """Parse a DualSense / DualShock 4 USB or Bluetooth HID report.

    hidapi may include or strip the report ID depending on the platform, so both
    layouts are accepted. DualSense USB analog triggers live at bytes 5-6 of the
    payload; DualShock 4 keeps buttons there and analog triggers later.
    """
    raw = list(data)
    if not raw:
        return None

    if product_id is not None:
        is_dualsense = product_id in SONY_DUALSENSE_PRODUCT_IDS
    else:
        is_dualsense = raw[0] in (0x01, 0x31)
    is_ds4 = product_id in SONY_DS4_PRODUCT_IDS if product_id is not None else False

    offset = 0
    if raw[0] == 0x31:
        # DualSense Bluetooth: report ID + extra header byte, then USB-like payload.
        offset = 2
        is_dualsense = True
    elif raw[0] == 0x11:
        # DualShock 4 Bluetooth.
        offset = 2
        is_ds4 = True
    elif raw[0] == 0x01 and len(raw) >= 48:
        # USB reports that still include the report ID are 64 bytes; do not treat a
        # fully-left stick (byte value 1) on a stripped report as a report ID.
        offset = 1
    payload = raw[offset:]
    if len(payload) < 9:
        return None

    lx = _u8_axis(payload[0])
    ly = _u8_axis(payload[1])
    rx = _u8_axis(payload[2])
    ry = _u8_axis(payload[3])

    if is_ds4 and not is_dualsense:
        buttons0 = payload[4]
        buttons1 = payload[5]
        l2 = (payload[7] / 255.0) if len(payload) > 8 else 0.0
        r2 = (payload[8] / 255.0) if len(payload) > 9 else 0.0
    else:
        # DualSense USB (and unknown Sony pads): analog triggers then buttons.
        l2 = payload[4] / 255.0
        r2 = payload[5] / 255.0
        buttons0 = payload[7]
        buttons1 = payload[8]

    return ParsedGamepadReport(
        left_x=lx,
        left_y=ly,
        right_x=rx,
        right_y=ry,
        trigger_left=l2,
        trigger_right=r2,
        l1=bool(buttons1 & 0x01),
        r1=bool(buttons1 & 0x02),
        south=bool(buttons0 & 0x20),  # Cross
        west=bool(buttons0 & 0x10),  # Square
        north=bool(buttons0 & 0x80),  # Triangle
    )


def classify_hid_report(
    data: list[int] | bytes, vendor_id: int | None, product_id: int | None
) -> str | None:
    """Return the parser name for a raw HID report, or None if it should be ignored."""
    if not data:
        return None
    if len(data) == 18 and data[0] == 0x20:
        return "xbox"
    is_sony = vendor_id == SONY_VENDOR_ID or product_id in SONY_GAMEPAD_PRODUCT_IDS
    if is_sony or data[0] in (0x01, 0x11, 0x31):
        if len(data) >= 8:
            return "sony"
    if len(data) >= 8:
        return "logitech"
    return None


def hid_device_is_gamepad(device: dict[str, Any]) -> bool:
    """True if an enumerated HID device looks like a gamepad rather than a mouse/headset."""
    usage_page = device.get("usage_page")
    usage = device.get("usage")
    vendor_id = device.get("vendor_id")
    product_id = device.get("product_id")
    product = (device.get("product_string") or "").lower()
    manufacturer = (device.get("manufacturer_string") or "").lower()

    if usage == HID_MOUSE_USAGE:
        return False

    if vendor_id == SONY_VENDOR_ID and product_id in SONY_GAMEPAD_PRODUCT_IDS:
        if usage_page == HID_GENERIC_DESKTOP_PAGE and usage in (HID_JOYSTICK_USAGE, HID_GAMEPAD_USAGE):
            return True
        # Windows hidapi sometimes reports usage 0 on the main DualSense interface.
        unknown_usages = (0x00, HID_JOYSTICK_USAGE, HID_GAMEPAD_USAGE)
        if usage_page in (HID_GENERIC_DESKTOP_PAGE, 0x00) and usage in unknown_usages:
            return True
        return False

    if usage_page == HID_GENERIC_DESKTOP_PAGE and usage in (HID_JOYSTICK_USAGE, HID_GAMEPAD_USAGE):
        return True

    haystack = f"{product} {manufacturer}"
    return any(token in haystack for token in GAMEPAD_NAME_TOKENS)


def hid_device_sort_key(device: dict[str, Any]) -> tuple[int, int]:
    """Prefer Game Pad usage over Joystick, then over unknown usage."""
    usage = device.get("usage")
    if usage == HID_GAMEPAD_USAGE:
        return (0, 0)
    if usage == HID_JOYSTICK_USAGE:
        return (1, 0)
    return (2, 0)


def match_hid_device_name(device: dict[str, Any], device_name: str) -> bool:
    needle = device_name.lower()
    product = (device.get("product_string") or "").lower()
    manufacturer = (device.get("manufacturer_string") or "").lower()
    vendor_id = device.get("vendor_id")
    product_id = device.get("product_id")
    ids = f"{vendor_id:04x}:{product_id:04x}" if vendor_id is not None and product_id is not None else ""
    return needle in product or needle in manufacturer or needle in ids


class InputController:
    """Base class for input controllers that generate motion deltas."""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0):
        """
        Initialize the controller.

        Args:
            x_step_size: Base movement step size in meters
            y_step_size: Base movement step size in meters
            z_step_size: Base movement step size in meters
        """
        self.x_step_size = x_step_size
        self.y_step_size = y_step_size
        self.z_step_size = z_step_size
        self.running = True
        self.episode_end_status = None  # None, "success", or "failure"
        self.intervention_flag = False
        self.open_gripper_command = False
        self.close_gripper_command = False
        self.wrist_roll_command = 0.0
        self.left_x = 0.0
        self.left_y = 0.0
        self.right_x = 0.0
        self.right_y = 0.0

    def start(self):
        """Start the controller and initialize resources."""
        pass

    def stop(self):
        """Stop the controller and release resources."""
        pass

    def get_deltas(self):
        """Get the current movement deltas (dx, dy, dz).

        Convention (SDL / DualSense / Xbox): stick up and left are negative.
        Left stick Y → forward/back (delta_x), left stick X → left/right (delta_y),
        right stick Y → up/down (delta_z).
        """
        delta_x = -self.left_y * self.x_step_size
        delta_y = -self.left_x * self.y_step_size
        delta_z = -self.right_y * self.z_step_size
        return delta_x, delta_y, delta_z

    def update(self):
        """Update controller state - call this once per frame."""
        pass

    def __enter__(self):
        """Support for use in 'with' statements."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure resources are released when exiting 'with' block."""
        self.stop()

    def get_episode_end_status(self):
        """
        Get the current episode end status.

        Returns:
            None if episode should continue, "success" or "failure" otherwise
        """
        status = self.episode_end_status
        self.episode_end_status = None  # Reset after reading
        return status

    def should_intervene(self):
        """Return True if intervention flag was set."""
        return self.intervention_flag

    def gripper_command(self):
        """Return the current gripper command."""
        if self.open_gripper_command == self.close_gripper_command:
            return "stay"
        elif self.open_gripper_command:
            return "open"
        elif self.close_gripper_command:
            return "close"

    def apply_parsed_report(self, report: ParsedGamepadReport, deadzone: float) -> None:
        """Copy a parsed HID/SDL report onto this controller's live state."""
        self.left_x = apply_deadzone(report.left_x, deadzone)
        self.left_y = apply_deadzone(report.left_y, deadzone)
        self.right_x = apply_deadzone(report.right_x, deadzone)
        self.right_y = apply_deadzone(report.right_y, deadzone)
        self.close_gripper_command = report.trigger_left > TRIGGER_PRESS_THRESHOLD
        self.open_gripper_command = report.trigger_right > TRIGGER_PRESS_THRESHOLD
        if report.l1 and not report.r1:
            self.wrist_roll_command = -1.0
        elif report.r1 and not report.l1:
            self.wrist_roll_command = 1.0
        else:
            self.wrist_roll_command = 0.0
        self.intervention_flag = report.r1
        if report.north:
            self.episode_end_status = TeleopEvents.SUCCESS
        elif report.west:
            self.episode_end_status = TeleopEvents.FAILURE
        elif report.south:
            self.episode_end_status = TeleopEvents.RERECORD_EPISODE
        else:
            self.episode_end_status = None


class KeyboardController(InputController):
    """Generate motion deltas from keyboard input."""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.key_states = {
            "forward_x": False,
            "backward_x": False,
            "forward_y": False,
            "backward_y": False,
            "forward_z": False,
            "backward_z": False,
            "quit": False,
            "success": False,
            "failure": False,
        }
        self.listener = None

    def start(self):
        """Start the keyboard listener."""
        if not pynput_can_capture():
            logging.warning(
                "Keyboard control is unavailable in this environment. pynput cannot capture keys "
                "on Wayland or headless machines, or on macOS without Accessibility / Input "
                "Monitoring permission. Keyboard motion will be inactive."
            )
            self.running = False
            return

        from pynput import keyboard

        def on_press(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = True
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = True
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = True
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = True
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = True
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = True
                elif key == keyboard.Key.esc:
                    self.key_states["quit"] = True
                    self.running = False
                    return False
                elif key == keyboard.Key.enter:
                    self.key_states["success"] = True
                    self.episode_end_status = TeleopEvents.SUCCESS
                elif key == keyboard.Key.backspace:
                    self.key_states["failure"] = True
                    self.episode_end_status = TeleopEvents.FAILURE
            except AttributeError:
                pass

        def on_release(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = False
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = False
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = False
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = False
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = False
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = False
                elif key == keyboard.Key.enter:
                    self.key_states["success"] = False
                elif key == keyboard.Key.backspace:
                    self.key_states["failure"] = False
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

        print("Keyboard controls:")
        print("  Arrow keys: Move in X-Y plane")
        print("  Shift and Shift_R: Move in Z axis")
        print("  Enter: End episode with SUCCESS")
        print("  Backspace: End episode with FAILURE")
        print("  ESC: Exit")

    def stop(self):
        """Stop the keyboard listener."""
        if self.listener and self.listener.is_alive():
            self.listener.stop()

    def get_deltas(self):
        """Get the current movement deltas from keyboard state."""
        delta_x = delta_y = delta_z = 0.0

        if self.key_states["forward_x"]:
            delta_x += self.x_step_size
        if self.key_states["backward_x"]:
            delta_x -= self.x_step_size
        if self.key_states["forward_y"]:
            delta_y += self.y_step_size
        if self.key_states["backward_y"]:
            delta_y -= self.y_step_size
        if self.key_states["forward_z"]:
            delta_z += self.z_step_size
        if self.key_states["backward_z"]:
            delta_z -= self.z_step_size

        return delta_x, delta_y, delta_z


def _load_sdl_controller_module():
    """Return pygame's SDL GameController module, or None if it is unavailable."""
    if pygame is None:
        return None
    try:
        from pygame._sdl2 import controller as sdl_controller
    except (ImportError, AttributeError):
        return None
    return sdl_controller


class GamepadController(InputController):
    """Generate motion deltas from gamepad input via pygame / SDL."""

    def __init__(
        self,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        deadzone=0.1,
        device_name: str | None = None,
    ):
        require_package("pygame", extra="gamepad")
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.device_name = device_name
        self.joystick = None
        self._sdl_controller = None
        self._sdl_module = None
        self._layout = XBOX_JOYSTICK_LAYOUT
        self.intervention_flag = False

    def start(self):
        """Initialize pygame and the gamepad."""
        pygame.init()
        pygame.joystick.init()
        self._sdl_module = _load_sdl_controller_module()
        if self._sdl_module is not None and not self._sdl_module.get_init():
            self._sdl_module.init()

        count = pygame.joystick.get_count()
        if count == 0:
            self.running = False
            raise RuntimeError("No gamepad detected. Please connect a gamepad and try again.")

        index = self._select_joystick_index(count)
        self.joystick = pygame.joystick.Joystick(index)
        self.joystick.init()
        name = self.joystick.get_name()
        self._layout = layout_for_device_name(name)

        if (
            self._sdl_module is not None
            and hasattr(self._sdl_module, "is_controller")
            and self._sdl_module.is_controller(index)
        ):
            self._sdl_controller = self._sdl_module.Controller(index)
            logger.info("Initialized gamepad via SDL GameController: %s", name)
        else:
            logger.info("Initialized gamepad via raw joystick: %s", name)

        print(f"Gamepad: {name}")
        print("Gamepad controls:")
        print("  Left stick up/down: shoulder_lift")
        print("  Left stick left/right: shoulder_pan")
        print("  Right stick up/down: elbow_flex")
        print("  Right stick left/right: wrist_flex")
        print("  L1 / R1 (LB / RB): wrist_roll")
        print("  L2 / R2 (LT / RT): gripper close / open")
        print("  Triangle / Y: End episode with SUCCESS")
        print("  Square / X: End episode with FAILURE")
        print("  Cross / A: Rerecord episode")

    def _select_joystick_index(self, count: int) -> int:
        if self.device_name is None:
            return 0
        needle = self.device_name.lower()
        available = []
        for i in range(count):
            name = pygame.joystick.Joystick(i).get_name()
            available.append(name)
            if needle in name.lower():
                return i
        self.running = False
        raise RuntimeError(
            f"No gamepad matching '{self.device_name}' found. Available: {available}. "
            "Use --teleop.device_name= to select one."
        )

    def stop(self):
        """Clean up pygame resources."""
        if self._sdl_controller is not None:
            try:
                self._sdl_controller.quit()
            except (AttributeError, pygame.error):
                pass
            self._sdl_controller = None
        if pygame.joystick.get_init():
            if self.joystick:
                self.joystick.quit()
            pygame.joystick.quit()
        pygame.quit()

    def update(self):
        """Process pygame events and read the current pad state."""
        pygame.event.pump()
        if self._sdl_controller is not None:
            self._update_from_sdl()
        elif self.joystick is not None:
            self._update_from_joystick()

    def _safe_axis(self, index: int | None) -> float:
        if index is None or self.joystick is None or index >= self.joystick.get_numaxes():
            return 0.0
        return float(self.joystick.get_axis(index))

    def _safe_button(self, index: int) -> bool:
        if self.joystick is None or index >= self.joystick.get_numbuttons():
            return False
        return bool(self.joystick.get_button(index))

    def _update_from_sdl(self):
        module = self._sdl_module
        ctrl = self._sdl_controller
        # pygame 2.5+ exposes CONTROLLER_AXIS_* on the module; fall back to SDL integers.
        axis_left_x = getattr(module, "CONTROLLER_AXIS_LEFTX", 0)
        axis_left_y = getattr(module, "CONTROLLER_AXIS_LEFTY", 1)
        axis_right_x = getattr(module, "CONTROLLER_AXIS_RIGHTX", 2)
        axis_right_y = getattr(module, "CONTROLLER_AXIS_RIGHTY", 3)
        axis_lt = getattr(module, "CONTROLLER_AXIS_TRIGGERLEFT", 4)
        axis_rt = getattr(module, "CONTROLLER_AXIS_TRIGGERRIGHT", 5)
        btn_a = getattr(module, "CONTROLLER_BUTTON_A", 0)
        btn_x = getattr(module, "CONTROLLER_BUTTON_X", 2)
        btn_y = getattr(module, "CONTROLLER_BUTTON_Y", 3)
        btn_l1 = getattr(module, "CONTROLLER_BUTTON_LEFTSHOULDER", 9)
        btn_r1 = getattr(module, "CONTROLLER_BUTTON_RIGHTSHOULDER", 10)

        self.apply_parsed_report(
            ParsedGamepadReport(
                left_x=ctrl.get_axis(axis_left_x) / 32767.0,
                left_y=ctrl.get_axis(axis_left_y) / 32767.0,
                right_x=ctrl.get_axis(axis_right_x) / 32767.0,
                right_y=ctrl.get_axis(axis_right_y) / 32767.0,
                trigger_left=ctrl.get_axis(axis_lt) / 32767.0,
                trigger_right=ctrl.get_axis(axis_rt) / 32767.0,
                l1=bool(ctrl.get_button(btn_l1)),
                r1=bool(ctrl.get_button(btn_r1)),
                south=bool(ctrl.get_button(btn_a)),
                west=bool(ctrl.get_button(btn_x)),
                north=bool(ctrl.get_button(btn_y)),
            ),
            self.deadzone,
        )

    def _update_from_joystick(self):
        layout = self._layout
        left_x = self._safe_axis(layout.left_x)
        left_y = self._safe_axis(layout.left_y)
        right_x = self._safe_axis(layout.right_x)
        right_y = self._safe_axis(layout.right_y)
        trigger_left = self._trigger_from_axis_or_button(layout.trigger_left, layout.button_l1)
        trigger_right = self._trigger_from_axis_or_button(layout.trigger_right, layout.button_r1)
        # Some pads expose L2/R2 only as buttons (indices often 6/7) when analog axes are missing.
        num_axes = self.joystick.get_numaxes() if self.joystick else 0
        if layout.trigger_left is not None and layout.trigger_left >= num_axes:
            trigger_left = 1.0 if self._safe_button(6) else 0.0
        if layout.trigger_right is not None and layout.trigger_right >= num_axes:
            trigger_right = 1.0 if self._safe_button(7) else 0.0

        self.apply_parsed_report(
            ParsedGamepadReport(
                left_x=left_x,
                left_y=left_y,
                right_x=right_x,
                right_y=right_y,
                trigger_left=trigger_left,
                trigger_right=trigger_right,
                l1=self._safe_button(layout.button_l1),
                r1=self._safe_button(layout.button_r1),
                south=self._safe_button(layout.button_south),
                west=self._safe_button(layout.button_west),
                north=self._safe_button(layout.button_north),
            ),
            self.deadzone,
        )

    def _trigger_from_axis_or_button(self, axis_index: int | None, _button_index: int) -> float:
        raw = self._safe_axis(axis_index)
        # Analog triggers are usually 0..1 (SDL) or -1..1 (DirectInput, rest = -1).
        if raw < 0:
            return (raw + 1.0) / 2.0
        return raw


class GamepadControllerHID(InputController):
    """Generate motion deltas from gamepad input using HIDAPI."""

    def __init__(
        self,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        deadzone=0.1,
        device_name: str | None = None,
    ):
        """
        Initialize the HID gamepad controller.

        Args:
            x_step_size: Base movement step size
            y_step_size: Base movement step size
            z_step_size: Base movement step size
            deadzone: Joystick deadzone to prevent drift
            device_name: Substring matched against product/manufacturer (or vid:pid).
        """
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.device_name = device_name
        self.device = None
        self.device_info = None
        self.vendor_id: int | None = None
        self.product_id: int | None = None
        self.buttons = {}

    def find_device(self):
        """Find a gamepad HID device by usage, Sony VID/PID, or name substring."""
        require_package("hidapi", extra="gamepad", import_name="hid")
        devices = hid.enumerate()
        gamepads = [d for d in devices if hid_device_is_gamepad(d)]
        gamepads.sort(key=hid_device_sort_key)

        if self.device_name is not None:
            for gp in gamepads:
                if match_hid_device_name(gp, self.device_name):
                    return gp
            available = [
                f"{d.get('manufacturer_string', '')} {d.get('product_string', '')}".strip() or hex_ids(d)
                for d in gamepads
            ]
            logging.error(
                "No gamepad matching '%s' found. Detected gamepads: %s",
                self.device_name,
                available if available else "(none)",
            )
            return None

        if gamepads:
            return gamepads[0]

        available = sorted(
            {
                f"{d.get('manufacturer_string', '')} {d.get('product_string', '')}".strip()
                for d in devices
                if d.get("product_string")
            }
        )
        logging.error(
            "No gamepad found (looked for HID Game Pad/Joystick usage and Sony DualSense/DS4). "
            "All HID devices: %s. Use --teleop.device_name= to match by product name.",
            available,
        )
        return None

    def start(self):
        """Connect to the gamepad using HIDAPI."""
        require_package("hidapi", extra="gamepad", import_name="hid")
        self.device_info = self.find_device()
        if not self.device_info:
            self.running = False
            raise RuntimeError(
                "No gamepad found. Check the connection, or pass --teleop.device_name= "
                "to select a device (product name or vid:pid)."
            )

        try:
            logging.info("Connecting to gamepad at path: %s", self.device_info["path"])
            self.device = hid.device()
            self.device.open_path(self.device_info["path"])
            self.device.set_nonblocking(1)
            self.vendor_id = self.device_info.get("vendor_id")
            self.product_id = self.device_info.get("product_id")

            manufacturer = self.device.get_manufacturer_string()
            product = self.device.get_product_string()
            logging.info(
                "Connected to %s %s (VID=%s PID=%s)",
                manufacturer,
                product,
                self.vendor_id,
                self.product_id,
            )

            logging.info("Gamepad controls (HID mode):")
            logging.info("  Left analog stick: shoulder_pan / shoulder_lift")
            logging.info("  Right analog stick: wrist_flex / elbow_flex")
            logging.info("  L1/R1: wrist_roll")
            logging.info("  L2/R2: gripper close / open")
            logging.info("  Triangle/Y: SUCCESS, Square/X: FAILURE, Cross/A: RERECORD")

        except OSError as e:
            self.running = False
            raise RuntimeError(
                f"Error opening gamepad: {e}. You might need to run with administrator privileges."
            ) from e

    def stop(self):
        """Close the HID device connection."""
        if self.device:
            self.device.close()
            self.device = None

    def update(self):
        """
        Read and process the latest gamepad data.
        hidapi often needs several reads before a stable report is available.
        """
        for _ in range(10):
            self._update()

    def _update(self):
        """Read and process the latest gamepad data."""
        if not self.device or not self.running:
            return

        try:
            data = self.device.read(64)
            if not data:
                return
            kind = classify_hid_report(data, self.vendor_id, self.product_id)
            report = None
            if kind == "xbox":
                report = parse_xbox_gip_report(data)
            elif kind == "sony":
                report = parse_sony_hid_report(data, self.product_id)
            elif kind == "logitech":
                report = parse_logitech_report(data)
            if report is not None:
                self.apply_parsed_report(report, self.deadzone)
        except OSError as e:
            logging.error("Error reading from gamepad: %s", e)


def hex_ids(device: dict[str, Any]) -> str:
    vendor_id = device.get("vendor_id")
    product_id = device.get("product_id")
    if vendor_id is None or product_id is None:
        return "unknown"
    return f"{vendor_id:04x}:{product_id:04x}"
