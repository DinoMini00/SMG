import torch
import os
import numpy as np
import gym
from algorithms.rl_utils import make_obs_grad_grid
import utils
import time
import shutil
from arguments import parse_args
from env.dmc_wrappers import make_env as make_dmc_env
from env.robot_wrappers import make_env as make_robot_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder
from torch.utils.tensorboard import SummaryWriter


def evaluate(env, agent, algorithm, num_episodes, L, step, test_env=False, eval_mode=None):
    episode_rewards = []
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_step = 0
        torch_obs = []
        torch_action = []
        while not done:
            with torch.no_grad():
                with utils.eval_mode(agent):
                    action = agent.select_action(obs)

                obs, reward, done, _ = env.step(action)
                episode_reward += reward
                # log in tensorboard 15th step
                if algorithm == 'sgsac' or 'smg' in algorithm:
                    if i == 0 and episode_step in [15, 16, 17, 18] and step > 0:
                        _obs = agent._obs_to_input(obs)
                        torch_obs.append(_obs)
                        torch_action.append(
                            torch.tensor(action).to(_obs.device).unsqueeze(0)
                        )
                        prefix = "eval" if eval_mode is None else eval_mode
                    if i == 0 and episode_step == 18 and step > 0:
                        _obs = torch.cat(torch_obs, 0)
                        _action = torch.cat(torch_action, 0)
                        if 'smg' in algorithm:
                            fore, back, recon_mask, recon_img = \
                                agent.predictor(_obs)
                            agent.log_image(_obs, step, prefix + "_eval_recon/raw")
                            agent.log_image(fore, step, prefix + "_eval_recon/fore")
                            agent.log_image(back, step, prefix + "_eval_recon/back")
                            agent.log_image(recon_mask*255, step, prefix + "_eval_recon/mask")
                            agent.log_image(recon_img, step, prefix + "_eval_recon/full")
                        else:
                            agent.log_tensorboard(_obs, _action, step, prefix=prefix)

                episode_step += 1

        episode_rewards.append(episode_reward)
    
    if L is not None:
        _test_env = f"_test_env_{eval_mode}" if test_env else ""
        # video.save(f"{step}{_test_env}.mp4")
        L.log(f"eval/episode_reward{_test_env}", np.mean(episode_rewards), step)

    return np.mean(episode_rewards)


def make_dmc_envs(args):
    env = make_dmc_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        mode="train",
        frame_stack=args.frame_stack
    )
    if args.eval_mode is not None:
        test_envs = []
        test_envs_mode = []

    if args.eval_mode in [
        "color_easy",
        "color_hard",
        "video_easy",
        "video_hard",
    ]:
        test_env = make_dmc_env(
            domain_name=args.domain_name,
            task_name=args.task_name,
            seed=args.seed + 42,
            episode_length=args.episode_length,
            action_repeat=args.action_repeat,
            image_size=args.image_size,
            mode=args.eval_mode,
            intensity=args.distracting_cs_intensity,
            frame_stack=args.frame_stack
        )

        test_envs.append(test_env)
        test_envs_mode.append(args.eval_mode)

    if args.eval_mode == "all":
        for eval_mode in ["color_easy", "color_hard", "video_easy", "video_hard"]:
            test_env = make_dmc_env(
                domain_name=args.domain_name,
                task_name=args.task_name,
                seed=args.seed + 42,
                episode_length=args.episode_length,
                action_repeat=args.action_repeat,
                image_size=args.image_size,
                mode=eval_mode,
                intensity=args.distracting_cs_intensity,
                frame_stack=args.frame_stack
            )
            test_envs.append(test_env)
            test_envs_mode.append(eval_mode)
    return env, test_envs, test_envs_mode


def make_robot_envs(args):
    if args.cameras == 0:
        cameras = ["third_person"]
    elif args.cameras == 1:
        cameras = ["first_person"]
    elif args.cameras == 2:
        cameras = ["third_person", "first_person"]
    else:
        raise Exception("Current Camera Pose Not Supported.")
    
    env = make_robot_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        episode_length=50,
        n_substeps=args.n_substeps,
        frame_stack=args.frame_stack,
        image_size=args.image_size,
        mode="train",
        cameras=cameras,  # ['third_person', 'first_person']
        observation_type=args.observation_type,
        action_space=args.action_space,
    )

    test_envs = []
    test_envs_mode = []
    for eval_mode in range(1, 6):
        test_env = make_robot_env(
            domain_name=args.domain_name,
            task_name=args.task_name,
            seed=args.seed,
            episode_length=50,
            n_substeps=args.n_substeps,
            frame_stack=args.frame_stack,
            image_size=args.image_size,
            mode="train",
            cameras=cameras,  # ['third_person', 'first_person']
            observation_type=args.observation_type,
            action_space=args.action_space,
            test=eval_mode,
        )
        test_envs.append(test_env)
        test_envs_mode.append(f"test_{eval_mode}")
    return env, test_envs, test_envs_mode


def make_adroit_envs(args):
    import dhm
    env = dhm.make_env_RRL(args.task_name)
    
    if args.eval_mode is not None:
        test_envs = []
        test_envs_mode = []

    if args.eval_mode in [
        "color_easy",
        "color_hard",
        "video_easy",
        "video_hard",
    ]:
        test_env = dhm.make_env_RRL(args.task_name,
                                    mode='test',
                                    level=args.eval_mode.replace('_', '-'))

        test_envs.append(test_env)
        test_envs_mode.append(args.eval_mode)

    if args.eval_mode == "all":
        for eval_mode in ["color_easy", "color_hard", "video_easy", "video_hard"]:
            test_env = dhm.make_env_RRL(args.task_name,
                                        mode='test',
                                        level=eval_mode.replace('_', '-'))
            test_envs.append(test_env)
            test_envs_mode.append(eval_mode)
    return env, test_envs, test_envs_mode


def main(args):
    home = os.environ["HOME"]
    os.environ["MJKEY_PATH"] = f"{home}/.mujoco/mujoco200_linux/bin/mjkey.txt"
    os.environ["MUJOCO_GL"] = "egl"
    # Set seed
    utils.set_seed_everywhere(args.seed)

    # Initialize environments
    gym.logger.set_level(40)
    if args.domain_name == 'robot':
        env, test_envs, test_envs_mode = make_robot_envs(args)
    elif args.domain_name == 'adroit':
        env, test_envs, test_envs_mode = make_adroit_envs(args)
    else:
        env, test_envs, test_envs_mode = make_dmc_envs(args)

    # Create working directory
    work_dir = os.path.join(
        args.log_dir,
        args.domain_name + "_" + args.task_name,
        args.algorithm,
        str(args.seed),
    )
    print("Working directory:", work_dir)
    if os.path.exists(work_dir):
        pass
        # shutil.rmtree(work_dir)
    assert not os.path.exists(
        os.path.join(work_dir, "train.log")
    ), "specified working directory already exists"
    utils.make_dir(work_dir)
    model_dir = utils.make_dir(os.path.join(work_dir, "model"))
    video_dir = utils.make_dir(os.path.join(work_dir, "video"))
    video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)
    utils.write_info(args, os.path.join(work_dir, "info.log"))


    # tensorboard
    tb_dir = os.path.join(
            args.log_dir,
            args.domain_name + "_" + args.task_name,
            args.algorithm,
            str(args.seed),
            "tensorboard",
        )
    print("tensorboard directory:", tb_dir)
    writer = SummaryWriter(tb_dir)
    L = Logger(log_dir=work_dir, writer=writer)

    # Prepare agent
    assert torch.cuda.is_available(), "must have cuda enabled"
    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.train_steps,
        batch_size=args.batch_size,
    )
    cropped_obs_shape = (
        3 * args.frame_stack,
        args.image_crop_size,
        args.image_crop_size,
    )
    print("Observations:", env.observation_space.shape)
    print("Cropped observations:", cropped_obs_shape)
    agent = make_agent(
        obs_shape=cropped_obs_shape, action_shape=env.action_space.shape, args=args, writer=writer
    )

    start_step, episode, episode_reward, done = 0, 0, 0, True
    start_time = time.time()
    for step in range(start_step, args.train_steps + 1):
        if done:
            if step > start_step:
                L.log("train/duration", time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            # Evaluate agent periodically
            if step != 0 and step % args.eval_freq == 0:
                print("Evaluating:", work_dir)
                L.log("eval/episode", episode, step)
                evaluate(env, agent,args.algorithm, args.eval_episodes, L, step)
                if test_envs is not None:
                    for test_env, test_env_mode in zip(test_envs, test_envs_mode):
                        evaluate(
                            test_env,
                            agent,
                            args.algorithm,
                            args.eval_episodes,
                            L,
                            step,
                            test_env=True,
                            eval_mode=test_env_mode,
                        )
                L.dump(step)

            # Save agent periodically
            if step > start_step and step % 250000 == 0:
                torch.save(
                    agent.actor.state_dict(), os.path.join(model_dir, f"actor_{step}.pt")
                )
                torch.save(
                    agent.critic.state_dict(), os.path.join(model_dir, f"critic_{step}.pt")
                )
                if args.algorithm == "sgsac":
                    torch.save(
                        agent.attribution_predictor.state_dict(), os.path.join(model_dir, f"attrib_predictor_{step}.pt")
                    )
                if args.algorithm == "smg":
                    torch.save(
                        agent.predictor.state_dict(), os.path.join(model_dir, f"predictor_{step}.pt")
                    )


            L.log("train/episode_reward", episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

            L.log("train/episode", episode, step)

        # Sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # Run training update
        if step >= args.init_steps:
            num_updates = args.init_steps if step == args.init_steps else 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        # Take step
        next_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
        replay_buffer.add(obs, action, reward, next_obs, done_bool)
        episode_reward += reward
        obs = next_obs

        episode_step += 1

    print("Completed training for", work_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
