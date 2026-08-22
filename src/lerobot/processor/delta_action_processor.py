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

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import PolicyAction, RobotAction, TransitionKey

from .pipeline import ActionProcessorStep, ProcessorStepRegistry, RobotActionProcessorStep

# Motor names for SO-100 / SO-101 (order matches the follower bus).
SO_FOLLOWER_MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


@ProcessorStepRegistry.register("map_tensor_to_delta_action_dict")
@dataclass
class MapTensorToDeltaActionDictStep(ActionProcessorStep):
    """
    Maps a flat action tensor from a policy to a structured delta action dictionary.

    This step is typically used after a policy outputs a continuous action vector.
    It decomposes the vector into named components for delta movements of the
    end-effector (x, y, z) and optionally the gripper.

    Attributes:
        use_gripper: If True, assumes the 4th element of the tensor is the
                     gripper action.
    """

    use_gripper: bool = True

    def action(self, action: PolicyAction) -> RobotAction:
        if not isinstance(action, PolicyAction):
            raise ValueError("Only PolicyAction is supported for this processor")

        if action.dim() > 1:
            action = action.squeeze(0)

        # TODO (maractingi): add rotation
        delta_action = {
            "delta_x": action[0].item(),
            "delta_y": action[1].item(),
            "delta_z": action[2].item(),
        }
        if self.use_gripper:
            delta_action["gripper"] = action[3].item()
        return delta_action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for axis in ["x", "y", "z"]:
            features[PipelineFeatureType.ACTION][f"delta_{axis}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        if self.use_gripper:
            features[PipelineFeatureType.ACTION]["gripper"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )
        return features


@ProcessorStepRegistry.register("map_delta_action_to_robot_action")
@dataclass
class MapDeltaActionToRobotActionStep(RobotActionProcessorStep):
    """
    Maps delta actions from teleoperators to robot target actions for inverse kinematics.

    This step converts a dictionary of delta movements (e.g., from a gamepad)
    into a target action format that includes an "enabled" flag and target
    end-effector positions. It also handles scaling and noise filtering.

    Attributes:
        position_scale: A factor to scale the delta position inputs.
        noise_threshold: The magnitude below which delta inputs are considered noise
                         and do not trigger an "enabled" state.
    """

    # Scale factors for delta movements
    position_scale: float = 1.0
    noise_threshold: float = 1e-3  # 1 mm threshold to filter out noise

    def action(self, action: RobotAction) -> RobotAction:
        # NOTE (maractingi): Action can be a dict from the teleop_devices or a tensor from the policy
        # TODO (maractingi): changing this target_xyz naming convention from the teleop_devices
        delta_x = action.pop("delta_x")
        delta_y = action.pop("delta_y")
        delta_z = action.pop("delta_z")
        delta_wz = action.pop("delta_wz", 0.0)
        gripper = action.pop("gripper")

        # Determine if the teleoperator is actively providing input
        # Consider enabled if any significant movement or rotation delta is detected
        position_magnitude = (delta_x**2 + delta_y**2 + delta_z**2) ** 0.5
        enabled = position_magnitude > self.noise_threshold or abs(delta_wz) > self.noise_threshold

        # Scale the deltas appropriately
        scaled_delta_x = delta_x * self.position_scale
        scaled_delta_y = delta_y * self.position_scale
        scaled_delta_z = delta_z * self.position_scale

        # Update action with robot target format
        action = {
            "enabled": enabled,
            "target_x": scaled_delta_x,
            "target_y": scaled_delta_y,
            "target_z": scaled_delta_z,
            "target_wx": 0.0,
            "target_wy": 0.0,
            "target_wz": float(delta_wz),
            "gripper_vel": float(gripper),
        }

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for axis in ["x", "y", "z"]:
            features[PipelineFeatureType.ACTION].pop(f"delta_{axis}", None)
        features[PipelineFeatureType.ACTION].pop("gripper", None)

        for feat in [
            "enabled",
            "target_x",
            "target_y",
            "target_z",
            "target_wx",
            "target_wy",
            "target_wz",
            "gripper_vel",
        ]:
            features[PipelineFeatureType.ACTION][f"{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features


@ProcessorStepRegistry.register("map_delta_action_to_joint_positions")
@dataclass
class MapDeltaActionToJointPositionsStep(RobotActionProcessorStep):
    """
    Maps delta teleop actions (gamepad / keyboard_ee) to joint position targets.

    Each incoming axis is applied as a per-frame increment on the matching joint,
    using the current robot observation as the reference. This avoids inverse
    kinematics and extra dependencies when driving an SO-100 / SO-101 follower.

    Axis mapping:
        delta_x  (left stick Y)  → shoulder_lift
        delta_y  (left stick X)  → shoulder_pan
        delta_z  (right stick Y) → elbow_flex
        delta_wx (right stick X) → wrist_flex
        delta_wz (L1/R1)         → wrist_roll
        gripper  (L2/R2)         → gripper (0=close, 1=stay, 2=open)
    """

    motor_names: list[str] = field(default_factory=lambda: list(SO_FOLLOWER_MOTOR_NAMES))
    joint_step_size: float = 3.0
    gripper_step_size: float = 5.0

    def action(self, action: RobotAction) -> RobotAction:
        raw_observation = self.transition.get(TransitionKey.OBSERVATION)
        if raw_observation is None:
            raise ValueError(
                "Robot observation is required to map delta teleop actions to joint positions."
            )
        observation = raw_observation.copy()

        delta_shoulder_pan = float(action.pop("delta_y", 0.0))
        delta_shoulder_lift = float(action.pop("delta_x", 0.0))
        delta_elbow_flex = float(action.pop("delta_z", 0.0))
        delta_wrist_flex = float(action.pop("delta_wx", 0.0))
        delta_wrist_roll = float(action.pop("delta_wz", 0.0))
        gripper = float(action.pop("gripper", 1.0))

        joint_deltas = {
            "shoulder_pan": delta_shoulder_pan,
            "shoulder_lift": delta_shoulder_lift,
            "elbow_flex": delta_elbow_flex,
            "wrist_flex": delta_wrist_flex,
            "wrist_roll": delta_wrist_roll,
        }

        result: RobotAction = {}
        for name in self.motor_names:
            key = f"{name}.pos"
            if key not in observation:
                raise ValueError(
                    f"Observation missing '{key}' required to map delta teleop to joint positions."
                )
            current_pos = float(observation[key])
            if name == "gripper":
                # GripperAction: CLOSE=0, STAY=1, OPEN=2. SO gripper range is 0 (closed) to 100 (open).
                if gripper == 0:
                    result[key] = max(current_pos - self.gripper_step_size, 0.0)
                elif gripper == 2:
                    result[key] = min(current_pos + self.gripper_step_size, 100.0)
                else:
                    result[key] = current_pos
            else:
                result[key] = current_pos + joint_deltas.get(name, 0.0) * self.joint_step_size

        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for key in ["delta_x", "delta_y", "delta_z", "delta_wx", "delta_wz", "gripper"]:
            features[PipelineFeatureType.ACTION].pop(key, None)

        for name in self.motor_names:
            features[PipelineFeatureType.ACTION][f"{name}.pos"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features
