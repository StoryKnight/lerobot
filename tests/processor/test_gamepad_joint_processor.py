"""Tests for MapDeltaActionToJointPositionsStep and make_default_processors."""

import pytest

from lerobot.configs import FeatureType, PipelineFeatureType
from lerobot.processor import (
    MapDeltaActionToJointPositionsStep,
    SO_FOLLOWER_MOTOR_NAMES,
    make_default_processors,
)
from lerobot.processor.converters import robot_action_observation_to_transition

MOTOR_NAMES = list(SO_FOLLOWER_MOTOR_NAMES)


def _make_observation(**overrides: float) -> dict[str, float]:
    obs = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 45.0,
        "elbow_flex.pos": -30.0,
        "wrist_flex.pos": 10.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 50.0,
    }
    obs.update(overrides)
    return obs


def _make_gamepad_action(
    delta_x: float = 0.0,
    delta_y: float = 0.0,
    delta_z: float = 0.0,
    delta_wx: float = 0.0,
    delta_wz: float = 0.0,
    gripper: float = 1.0,
) -> dict[str, float]:
    return {
        "delta_x": delta_x,
        "delta_y": delta_y,
        "delta_z": delta_z,
        "delta_wx": delta_wx,
        "delta_wz": delta_wz,
        "gripper": gripper,
    }


def _run_step(
    step: MapDeltaActionToJointPositionsStep,
    action: dict[str, float],
    observation: dict[str, float],
) -> dict[str, float]:
    transition = robot_action_observation_to_transition((action, observation))
    result_transition = step(transition)
    return result_transition["action"]


class TestMapDeltaActionToJointPositionsStep:
    def test_no_input_preserves_positions(self):
        step = MapDeltaActionToJointPositionsStep(motor_names=MOTOR_NAMES)
        obs = _make_observation()
        result = _run_step(step, _make_gamepad_action(), obs)

        for name in MOTOR_NAMES:
            assert result[f"{name}.pos"] == pytest.approx(obs[f"{name}.pos"])

    def test_left_stick_y_moves_shoulder_lift(self):
        step = MapDeltaActionToJointPositionsStep(motor_names=MOTOR_NAMES, joint_step_size=3.0)
        result = _run_step(step, _make_gamepad_action(delta_x=1.0), _make_observation())

        assert result["shoulder_lift.pos"] == pytest.approx(45.0 + 3.0)
        assert result["shoulder_pan.pos"] == pytest.approx(0.0)
        assert result["elbow_flex.pos"] == pytest.approx(-30.0)

    def test_left_stick_x_moves_shoulder_pan(self):
        step = MapDeltaActionToJointPositionsStep(motor_names=MOTOR_NAMES, joint_step_size=3.0)
        result = _run_step(step, _make_gamepad_action(delta_y=-0.5), _make_observation())

        assert result["shoulder_pan.pos"] == pytest.approx(-0.5 * 3.0)

    def test_right_stick_y_moves_elbow_flex(self):
        step = MapDeltaActionToJointPositionsStep(motor_names=MOTOR_NAMES, joint_step_size=2.0)
        result = _run_step(step, _make_gamepad_action(delta_z=0.8), _make_observation())

        assert result["elbow_flex.pos"] == pytest.approx(-30.0 + 0.8 * 2.0)

    def test_right_stick_x_moves_wrist_flex(self):
        step = MapDeltaActionToJointPositionsStep(motor_names=MOTOR_NAMES, joint_step_size=3.0)
        result = _run_step(step, _make_gamepad_action(delta_wx=-1.0), _make_observation())

        assert result["wrist_flex.pos"] == pytest.approx(10.0 + (-1.0 * 3.0))

    def test_bumpers_move_wrist_roll(self):
        step = MapDeltaActionToJointPositionsStep(motor_names=MOTOR_NAMES, joint_step_size=3.0)
        obs = _make_observation()

        result = _run_step(step, _make_gamepad_action(delta_wz=1.0), obs)
        assert result["wrist_roll.pos"] == pytest.approx(3.0)

        result = _run_step(step, _make_gamepad_action(delta_wz=-1.0), obs)
        assert result["wrist_roll.pos"] == pytest.approx(-3.0)

    def test_gripper_close_decreases_position(self):
        step = MapDeltaActionToJointPositionsStep(motor_names=MOTOR_NAMES, gripper_step_size=5.0)
        result = _run_step(step, _make_gamepad_action(gripper=0.0), _make_observation())

        assert result["gripper.pos"] == pytest.approx(45.0)

    def test_gripper_open_increases_position(self):
        step = MapDeltaActionToJointPositionsStep(motor_names=MOTOR_NAMES, gripper_step_size=5.0)
        result = _run_step(step, _make_gamepad_action(gripper=2.0), _make_observation())

        assert result["gripper.pos"] == pytest.approx(55.0)

    def test_gripper_stay(self):
        step = MapDeltaActionToJointPositionsStep(motor_names=MOTOR_NAMES)
        result = _run_step(step, _make_gamepad_action(gripper=1.0), _make_observation())

        assert result["gripper.pos"] == pytest.approx(50.0)

    def test_gripper_clamps_at_zero(self):
        step = MapDeltaActionToJointPositionsStep(motor_names=MOTOR_NAMES, gripper_step_size=10.0)
        obs = _make_observation(**{"gripper.pos": 3.0})
        result = _run_step(step, _make_gamepad_action(gripper=0.0), obs)

        assert result["gripper.pos"] == pytest.approx(0.0)

    def test_gripper_clamps_at_hundred(self):
        step = MapDeltaActionToJointPositionsStep(motor_names=MOTOR_NAMES, gripper_step_size=10.0)
        obs = _make_observation(**{"gripper.pos": 97.0})
        result = _run_step(step, _make_gamepad_action(gripper=2.0), obs)

        assert result["gripper.pos"] == pytest.approx(100.0)

    def test_multiple_axes_simultaneously(self):
        step = MapDeltaActionToJointPositionsStep(motor_names=MOTOR_NAMES, joint_step_size=2.0)
        action = _make_gamepad_action(delta_x=1.0, delta_y=-0.5, delta_z=0.3)
        result = _run_step(step, action, _make_observation())

        assert result["shoulder_lift.pos"] == pytest.approx(45.0 + 2.0)
        assert result["shoulder_pan.pos"] == pytest.approx(-1.0)
        assert result["elbow_flex.pos"] == pytest.approx(-30.0 + 0.6)

    def test_outputs_all_motor_names(self):
        step = MapDeltaActionToJointPositionsStep(motor_names=MOTOR_NAMES)
        result = _run_step(step, _make_gamepad_action(), _make_observation())

        for name in MOTOR_NAMES:
            assert f"{name}.pos" in result

    def test_missing_observation_raises(self):
        step = MapDeltaActionToJointPositionsStep(motor_names=MOTOR_NAMES)
        transition = robot_action_observation_to_transition((_make_gamepad_action(), None))
        with pytest.raises(ValueError, match="observation"):
            step(transition)

    def test_missing_joint_key_raises(self):
        step = MapDeltaActionToJointPositionsStep(motor_names=MOTOR_NAMES)
        obs = {"shoulder_pan.pos": 0.0}
        with pytest.raises(ValueError, match="shoulder_lift.pos"):
            _run_step(step, _make_gamepad_action(), obs)

    def test_transform_features_removes_deltas_adds_motor_pos(self, policy_feature_factory):
        step = MapDeltaActionToJointPositionsStep(motor_names=MOTOR_NAMES)
        features = {
            PipelineFeatureType.ACTION: {
                "delta_x": policy_feature_factory(FeatureType.ACTION, (1,)),
                "delta_y": policy_feature_factory(FeatureType.ACTION, (1,)),
                "delta_z": policy_feature_factory(FeatureType.ACTION, (1,)),
                "delta_wx": policy_feature_factory(FeatureType.ACTION, (1,)),
                "delta_wz": policy_feature_factory(FeatureType.ACTION, (1,)),
                "gripper": policy_feature_factory(FeatureType.ACTION, (1,)),
            },
            PipelineFeatureType.OBSERVATION: {},
        }

        out = step.transform_features(features)

        for key in ["delta_x", "delta_y", "delta_z", "delta_wx", "delta_wz", "gripper"]:
            assert key not in out[PipelineFeatureType.ACTION]
        for name in MOTOR_NAMES:
            assert f"{name}.pos" in out[PipelineFeatureType.ACTION]


class _FakeConfig:
    def __init__(self, config_type: str):
        self._type = config_type

    @property
    def type(self) -> str:
        return self._type


class TestMakeProcessors:
    def test_gamepad_so_follower_returns_joint_pipeline(self):
        teleop_proc, _, _ = make_default_processors(_FakeConfig("gamepad"), _FakeConfig("so101_follower"))

        assert len(teleop_proc.steps) == 1
        assert isinstance(teleop_proc.steps[0], MapDeltaActionToJointPositionsStep)

    def test_keyboard_ee_so_follower_returns_joint_pipeline(self):
        teleop_proc, _, _ = make_default_processors(
            _FakeConfig("keyboard_ee"), _FakeConfig("so100_follower")
        )

        assert isinstance(teleop_proc.steps[0], MapDeltaActionToJointPositionsStep)

    def test_leader_follower_returns_default_pipeline(self):
        result = make_default_processors(_FakeConfig("so101_leader"), _FakeConfig("so101_follower"))
        default = make_default_processors()

        assert len(result[0].steps) == len(default[0].steps)
        assert type(result[0].steps[0]) is type(default[0].steps[0])

    def test_gamepad_non_so_robot_returns_default_pipeline(self):
        result = make_default_processors(_FakeConfig("gamepad"), _FakeConfig("koch_follower"))
        default = make_default_processors()

        assert type(result[0].steps[0]) is type(default[0].steps[0])

    def test_so_motor_names_constant(self):
        assert SO_FOLLOWER_MOTOR_NAMES == [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]

    def test_joint_pipeline_end_to_end(self):
        teleop_proc, robot_proc, _ = make_default_processors(
            _FakeConfig("gamepad"), _FakeConfig("so101_follower")
        )
        obs = _make_observation()
        action = _make_gamepad_action(delta_x=1.0, gripper=2.0)

        teleop_result = teleop_proc((action, obs))
        assert "shoulder_lift.pos" in teleop_result
        assert teleop_result["shoulder_lift.pos"] == pytest.approx(48.0)
        assert teleop_result["gripper.pos"] == pytest.approx(55.0)

        robot_result = robot_proc((teleop_result, obs))
        assert robot_result == teleop_result
