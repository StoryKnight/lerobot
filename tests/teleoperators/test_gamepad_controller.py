"""Tests for gamepad HID parsers and axis mapping (no hardware required)."""

import struct

import pytest

from lerobot.teleoperators.gamepad.gamepad_utils import (
    DUALSENSE_JOYSTICK_LAYOUT,
    SONY_VENDOR_ID,
    XBOX_JOYSTICK_LAYOUT,
    InputController,
    apply_deadzone,
    classify_hid_report,
    hid_device_is_gamepad,
    layout_for_device_name,
    parse_logitech_report,
    parse_sony_hid_report,
    parse_xbox_gip_report,
)
from lerobot.teleoperators.utils import TeleopEvents


class TestInputController:
    def test_default_attributes(self):
        ctrl = InputController()
        assert ctrl.wrist_roll_command == 0.0
        assert ctrl.right_x == 0.0
        assert ctrl.open_gripper_command is False
        assert ctrl.close_gripper_command is False

    def test_gripper_command_stay(self):
        assert InputController().gripper_command() == "stay"

    def test_gripper_command_open(self):
        ctrl = InputController()
        ctrl.open_gripper_command = True
        assert ctrl.gripper_command() == "open"

    def test_gripper_command_close(self):
        ctrl = InputController()
        ctrl.close_gripper_command = True
        assert ctrl.gripper_command() == "close"

    def test_left_stick_up_gives_positive_delta_x(self):
        ctrl = InputController()
        ctrl.left_y = -1.0
        dx, dy, dz = ctrl.get_deltas()
        assert dx == pytest.approx(1.0)
        assert dy == 0.0
        assert dz == 0.0

    def test_left_stick_right_gives_negative_delta_y(self):
        ctrl = InputController()
        ctrl.left_x = 1.0
        _, dy, _ = ctrl.get_deltas()
        assert dy == pytest.approx(-1.0)


class TestLayoutForDeviceName:
    def test_dualsense_names(self):
        for name in (
            "DualSense Wireless Controller",
            "PS5 Controller",
            "Sony Interactive Entertainment Wireless Controller",
            "Wireless Controller",
        ):
            assert layout_for_device_name(name) is DUALSENSE_JOYSTICK_LAYOUT

    def test_xbox_names(self):
        for name in ("Xbox One Controller", "Xbox 360 Controller", "Generic USB Joystick"):
            assert layout_for_device_name(name) is XBOX_JOYSTICK_LAYOUT


class TestApplyDeadzone:
    def test_filters_small_values(self):
        assert apply_deadzone(0.05, 0.1) == 0.0

    def test_keeps_large_values(self):
        assert apply_deadzone(0.5, 0.1) == pytest.approx(0.5)


def _build_xbox_report(
    *,
    lx: int = 0,
    ly: int = 0,
    rx: int = 0,
    ry: int = 0,
    lt: int = 0,
    rt: int = 0,
    buttons0: int = 0,
    buttons1: int = 0,
) -> list[int]:
    buf = bytearray(18)
    buf[0] = 0x20
    struct.pack_into("<H", buf, 6, lt)
    struct.pack_into("<H", buf, 8, rt)
    struct.pack_into("<h", buf, 10, lx)
    struct.pack_into("<h", buf, 12, ly)
    struct.pack_into("<h", buf, 14, rx)
    struct.pack_into("<h", buf, 16, ry)
    buf[4] = buttons0
    buf[5] = buttons1
    return list(buf)


def _build_dualsense_usb_report(
    *,
    lx: int = 128,
    ly: int = 128,
    rx: int = 128,
    ry: int = 128,
    l2: int = 0,
    r2: int = 0,
    buttons0: int = 0,
    buttons1: int = 0,
    include_report_id: bool = True,
) -> list[int]:
    payload = [lx, ly, rx, ry, l2, r2, 0, buttons0, buttons1] + [0] * 54
    if include_report_id:
        return [0x01, *payload]
    return payload


class TestXboxHIDParsing:
    def test_sticks_at_rest(self):
        report = parse_xbox_gip_report(_build_xbox_report())
        assert report is not None
        assert report.left_x == 0.0
        assert report.left_y == 0.0
        assert report.right_x == 0.0
        assert report.right_y == 0.0

    def test_left_stick_full_up_is_negative_y(self):
        report = parse_xbox_gip_report(_build_xbox_report(ly=32767))
        assert report is not None
        assert report.left_y == pytest.approx(-32767 / 32768.0, abs=0.01)

    def test_triggers_and_bumpers(self):
        report = parse_xbox_gip_report(_build_xbox_report(lt=500, buttons1=(1 << 4)))
        assert report is not None
        assert report.trigger_left == pytest.approx(500 / 1023.0, abs=0.01)
        assert report.l1 is True

    def test_y_button(self):
        report = parse_xbox_gip_report(_build_xbox_report(buttons0=(1 << 7)))
        assert report is not None
        assert report.north is True

    def test_wrong_header_returns_none(self):
        assert parse_xbox_gip_report([0x01] * 18) is None


class TestDualSenseHIDParsing:
    def test_sticks_at_rest(self):
        report = parse_sony_hid_report(_build_dualsense_usb_report(), product_id=0x0CE6)
        assert report is not None
        assert report.left_x == pytest.approx(0.0)
        assert report.left_y == pytest.approx(0.0)

    def test_left_stick_up(self):
        # DualSense: 0 = up, 255 = down (SDL-style already).
        report = parse_sony_hid_report(_build_dualsense_usb_report(ly=0), product_id=0x0CE6)
        assert report is not None
        assert report.left_y == pytest.approx(-1.0)

    def test_left_stick_right(self):
        report = parse_sony_hid_report(_build_dualsense_usb_report(lx=255), product_id=0x0CE6)
        assert report is not None
        assert report.left_x == pytest.approx((255 - 128) / 128.0)

    def test_triggers(self):
        report = parse_sony_hid_report(_build_dualsense_usb_report(l2=200, r2=10), product_id=0x0CE6)
        assert report is not None
        assert report.trigger_left == pytest.approx(200 / 255.0)
        assert report.trigger_right == pytest.approx(10 / 255.0)

    def test_face_buttons(self):
        # Square=bit4, Cross=bit5, Triangle=bit7
        report = parse_sony_hid_report(_build_dualsense_usb_report(buttons0=0x20), product_id=0x0CE6)
        assert report is not None
        assert report.south is True

        report = parse_sony_hid_report(_build_dualsense_usb_report(buttons0=0x80), product_id=0x0CE6)
        assert report is not None
        assert report.north is True

    def test_shoulders(self):
        report = parse_sony_hid_report(_build_dualsense_usb_report(buttons1=0x01), product_id=0x0CE6)
        assert report is not None
        assert report.l1 is True
        assert report.r1 is False

    def test_stripped_report_id(self):
        report = parse_sony_hid_report(
            _build_dualsense_usb_report(ly=0, include_report_id=False), product_id=0x0CE6
        )
        assert report is not None
        assert report.left_y == pytest.approx(-1.0)

    def test_bluetooth_report(self):
        usb = _build_dualsense_usb_report(lx=255, include_report_id=False)
        bt = [0x31, 0x01, *usb]
        report = parse_sony_hid_report(bt, product_id=0x0CE6)
        assert report is not None
        assert report.left_x == pytest.approx((255 - 128) / 128.0)

    def test_full_pipeline_stick_up_is_forward(self):
        ctrl = InputController()
        report = parse_sony_hid_report(_build_dualsense_usb_report(ly=0), product_id=0x0CE6)
        ctrl.apply_parsed_report(report, deadzone=0.05)
        dx, _, _ = ctrl.get_deltas()
        assert dx > 0.9

    def test_r2_opens_gripper(self):
        ctrl = InputController()
        report = parse_sony_hid_report(_build_dualsense_usb_report(r2=255), product_id=0x0CE6)
        ctrl.apply_parsed_report(report, deadzone=0.1)
        assert ctrl.gripper_command() == "open"

    def test_l1_rolls_left(self):
        ctrl = InputController()
        report = parse_sony_hid_report(_build_dualsense_usb_report(buttons1=0x01), product_id=0x0CE6)
        ctrl.apply_parsed_report(report, deadzone=0.1)
        assert ctrl.wrist_roll_command == -1.0
        assert ctrl.episode_end_status is None

    def test_triangle_success(self):
        ctrl = InputController()
        report = parse_sony_hid_report(_build_dualsense_usb_report(buttons0=0x80), product_id=0x0CE6)
        ctrl.apply_parsed_report(report, deadzone=0.1)
        assert ctrl.episode_end_status == TeleopEvents.SUCCESS


class TestClassifyAndDetect:
    def test_classify_xbox(self):
        assert classify_hid_report(_build_xbox_report(), vendor_id=0x045E, product_id=0x0B13) == "xbox"

    def test_classify_sony(self):
        data = _build_dualsense_usb_report()
        assert classify_hid_report(data, vendor_id=SONY_VENDOR_ID, product_id=0x0CE6) == "sony"

    def test_classify_logitech(self):
        data = [0, 128, 128, 128, 128, 0, 0, 0]
        assert classify_hid_report(data, vendor_id=0x046D, product_id=0xC216) == "logitech"

    def test_hid_device_is_gamepad_dualsense(self):
        device = {
            "vendor_id": SONY_VENDOR_ID,
            "product_id": 0x0CE6,
            "usage_page": 0x01,
            "usage": 0x05,
            "product_string": "",
            "manufacturer_string": "",
        }
        assert hid_device_is_gamepad(device) is True

    def test_hid_device_skips_touchpad_mouse(self):
        device = {
            "vendor_id": SONY_VENDOR_ID,
            "product_id": 0x0CE6,
            "usage_page": 0x01,
            "usage": 0x02,
            "product_string": "DualSense",
            "manufacturer_string": "Sony",
        }
        assert hid_device_is_gamepad(device) is False

    def test_logitech_parser_idle(self):
        report = parse_logitech_report([0, 128, 128, 128, 128, 0, 0, 0])
        assert report is not None
        assert report.left_x == pytest.approx(0.0)
