#TODO: @shuaiyy create base for unitree g1

from legged_gym.envs.base.base_config import BaseConfig
from legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfg

class G1RoughCfg(BaseConfig):
    class human:
        delay = 0.0 # delay in seconds
        # freq = 10
        freq = 30
        resample_on_env_reset = True
        # filename = 'ACCAD_walk_10fps.npy'
        filename = 'twist2_dataset.yaml'
        
    class env:
        # num_envs = 2048
        num_envs = 4
        num_dofs = 29
        num_observations = 8 + 4 * num_dofs # [orn_rp, ang_vel, commands, dof_pos, dof_vel, actions, delayed_target_jt]
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 29
        env_spacing = 5.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

        action_delay = 1  # -1 for no delay
        obs_context_len = 8

    class terrain:
        mesh_type = "trimesh" # none, plane, heightfield or trimesh
        height = [0, 0.00]
        horizontal_scale = 0.1 # [m]

        hf2mesh_method = "grid"  # grid or fast
        max_error = 0.1 # for fast
        max_error_camera = 2

        y_range = [-0.4, 0.4]
        
        edge_width_thresh = 0.05
        horizontal_scale = 0.15 # [m] influence computation time by a lot
        horizontal_scale_camera = 0.1
        vertical_scale = 0.005 # [m]
        border_size = 5 # [m]
        height = [0, 0.0]
        simplify_grid = False
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075


        ##refereced from twiat2
        curriculum = False

        all_vertical = False
        no_flat = True
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = False
        measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]
        measure_horizontal_noise = 0.0

        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 18.
        terrain_width = 4
        num_rows= 10 # number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols = 40 # number of terrain cols (types)
        
        terrain_dict = {"smooth slope": 0., 
                        "rough slope up": 0.0,
                        "rough slope down": 0.0,
                        "rough stairs up": 0., 
                        "rough stairs down": 0., 
                        "discrete": 0., 
                        "stepping stones": 0.0,
                        "gaps": 0., 
                        "smooth flat": 0,
                        "pit": 0.0,
                        "wall": 0.0,
                        "platform": 0.,
                        "large stairs up": 0.,
                        "large stairs down": 0.,
                        "parkour": 0.,
                        "parkour_hurdle": 0.,
                        "parkour_flat": 0.05,
                        "parkour_step": 0.0,
                        "parkour_gap": 0.0,
                        "demo": 0.0,}
        terrain_proportions = list(terrain_dict.values())
        
        # trimesh only:
        slope_treshold = 1.5# slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = True

        num_goals = 8
    
    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [0.9, 0.9] # min max [m/s]
            lin_vel_y = [0, 0]   # min max [m/s]
            ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [0, 0]

    class init_state:
        pos = [0, 0, 1.0]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {
            'left_hip_pitch_joint': -0.2,
            'left_hip_roll_joint': 0.0,
            'left_hip_yaw_joint': 0.0,
            'left_knee_joint': 0.4,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0.0,
            
            'right_hip_pitch_joint': -0.2,
            'right_hip_roll_joint': 0.0,
            'right_hip_yaw_joint': 0.0,
            'right_knee_joint': 0.4,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0.0,
            
            'waist_yaw_joint': 0.0,
            'waist_roll_joint': 0.0,
            'waist_pitch_joint': 0.0,
            
            'left_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.4,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 1.2,
            'left_wrist_roll_joint': 0.0,
            'left_wrist_pitch_joint': 0.0,
            'left_wrist_yaw_joint': 0.0,
            
            'right_shoulder_pitch_joint': 0.0,
            'right_shoulder_roll_joint': -0.4,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_joint': 1.2,
            'right_wrist_roll_joint': 0.0,
            'right_wrist_pitch_joint': 0.0,
            'right_wrist_yaw_joint': 0.0,
        }
    
    class control(HumanoidMimicCfg.control):
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     'waist': 150,
                     'shoulder': 40,
                     'elbow': 40,
                     'wrist': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     'waist': 4,
                     'shoulder': 5,
                     'elbow': 5,
                     'wrist': 5,
                     }  # [N*m/rad]  # [N*m*s/rad]
        
        action_scale = 0.5 # Scales the policy action before converting it to a joint target: target_angle = default_joint_angle + action_scale * action
        clip_actions = True # if true actions are clipped to [-action_scale, action_scale]
        decimation = 10 # (10 in twist2, but 4 in humanplus)decimation is the number of simulator timesteps for which the same policy action is reused
    
    class domain_rand:
        domain_rand_general = True # manually set this, setting from parser does not work;
        
        randomize_gravity = (True and domain_rand_general)
        gravity_rand_interval_s = 4
        gravity_range = (-0.1, 0.1)
        
        randomize_friction = (True and domain_rand_general)
        friction_range = [0.1, 2.]
        
        randomize_base_mass = (True and domain_rand_general)
        added_mass_range = [-3., 3]
        
        randomize_base_com = (True and domain_rand_general)
        added_com_range = [-0.05, 0.05]
        
        push_robots = (True and domain_rand_general)
        push_interval_s = 4
        max_push_vel_xy = 1.0
        
        push_end_effector = (True and domain_rand_general)
        push_end_effector_interval_s = 2
        max_push_force_end_effector = 20.0

        randomize_motor = (True and domain_rand_general)
        motor_strength_range = [0.8, 1.2]

        action_delay = (True and domain_rand_general)
        action_buf_len = 8

    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 0.1
            tracking_ang_vel = 0.1
            lin_vel_z = -0
            ang_vel_xy = -0
            orientation = -0.
            torques = -0.0
            dof_vel = -0.
            dof_acc = -0.
            base_height = -0. 
            feet_air_time = 0.
            collision = -0.
            feet_stumble = -0.0 
            action_rate = -0.
            stand_still = -0.
            dof_pos_limits = -0.0
            target_jt = 1

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized

    class termination:
        r_threshold = 0.5
        p_threshold = 0.5
        z_threshold = 0.85

    class normalization(HumanoidMimicCfg.normalization):
        class obs_scales:
            orn = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            imu = 0.5
        commands_scale = [1., 1., 1., 1.]
        clip_actions = 5.0

    class asset(HumanoidMimicCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/../../assets/g1/g1_custom_collision.urdf'
        file = '{LEGGED_GYM_ROOT_DIR}/../../assets/g1/g1_custom_collision_29dof.urdf'
        name = 'g1'
        # for both joint and link name
        torso_name: str = 'pelvis'  # humanoid pelvis part
        chest_name: str = 'imu_in_torso'  # humanoid chest part

        # for link name
        thigh_name: str = 'hip'
        shank_name: str = 'knee'
        foot_name: str = 'ankle_roll_link' # matches left/right ankle roll links in the G1 URDF
        waist_name: list = ['torso_link', 'waist_roll_link', 'waist_yaw_link']
        upper_arm_name: str = 'shoulder_roll_link'
        lower_arm_name: str = 'elbow_link'
        hand_name: str = 'hand'

        feet_bodies = ['left_ankle_roll_link', 'right_ankle_roll_link']
        n_lower_body_dofs: int = 12

        penalize_contacts_on = ["pelvis", "shoulder", "elbow", "hip", "knee"]
        terminate_after_contacts_on = ['torso_link']
        
        # ========================= Inertia =========================
        # shoulder, elbow, and ankle: 0.139 * 1e-4 * 16**2 + 0.017 * 1e-4 * (46/18 + 1)**2 + 0.169 * 1e-4 = 0.003597
        # waist, hip pitch & yaw: 0.489 * 1e-4 * 14.3**2 + 0.098 * 1e-4 * 4.5**2 + 0.533 * 1e-4 = 0.0103
        # knee, hip roll: 0.489 * 1e-4 * 22.5**2 + 0.109 * 1e-4 * 4.5**2 + 0.738 * 1e-4 = 0.0251
        # wrist: 0.068 * 1e-4 * 25**2 = 0.00425
        
        dof_armature = [0.0103, 0.0251, 0.0103, 0.0251, 0.003597, 0.003597] * 2 + [0.0103] * 3 + [0.003597] * 14
        
        # dof_armature = [0.0, 0.0, 0.0, 0.0, 0.0, 0.001] * 2 + [0.0] * 3 + [0.0] * 8
        
        # ========================= Inertia =========================
        
        collapse_fixed_joints = False
    
    
    class noise(HumanoidMimicCfg.noise):
        add_noise = True
        noise_increasing_steps = 3000
        class noise_scales:
            orn = 0.05
            dof_pos = 0.01
            dof_vel = 0.1
            lin_vel = 0.1
            ang_vel = 0.1
            gravity = 0.05
            imu = 0.1
    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim(HumanoidMimicCfg.sim):
        dt = 0.002

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 8
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class G1RoughCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        # actor_hidden_dims = [512, 256, 128]
        # critic_hidden_dims = [512, 256, 128]
        # activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0 # Weight of the critic/value loss term.
        use_clipped_value_loss = True # Clip critic updates for more stable value learning.
        clip_param = 0.2 # PPO clip range that limits how much the policy can change per update.
        entropy_coef = 1e-5 # Exploration bonus weight; higher means more random actions.
        num_learning_epochs = 2 # Number of passes over the collected rollout per PPO update.
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-4 # Optimizer step size for backprop.
        schedule = 'fixed' # Learning-rate schedule; fixed keeps it constant.
        gamma = 0.99 # Discount factor for future rewards.
        lam = 0.95 # GAE parameter controlling bias vs variance in advantage estimates.
        desired_kl = 0.01 # Target KL used to monitor/control update size.
        max_grad_norm = 1. # Gradient clipping threshold for training stability.

    class runner:
        policy_class_name = 'ActorCriticTransformer'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 32 # per iteration
        max_iterations = 15000 # number of policy updates

        # logging
        save_interval = 1000 # check for potential saves every this many iterations
        experiment_name = 'rough_g1'
        run_name = 'g1_debug' 
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
