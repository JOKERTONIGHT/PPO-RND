from torch import device
import os
from Common.utils import *
import time
import gymnasium as gym
import datetime
from PIL import Image
import imageio

# Import and register Atari environments
try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    print("Warning: Could not import ale_py")


class Play:
    def __init__(self, env, agent, checkpoint, max_episode=1, save_gif=False, gif_fps=30):
        # 指定render_mode='rgb_array'以便记录游戏画面，或'human'用于可视化
        render_mode = 'rgb_array' if save_gif else 'human'
        self.env = make_atari(env, 4500, sticky_action=False, render_mode=render_mode)
        self.max_episode = max_episode
        self.agent = agent
        self.agent.set_from_checkpoint(checkpoint)
        self.agent.set_to_eval_mode()
        self.device = device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_gif = save_gif
        self.gif_fps = gif_fps
        
        # 创建结果目录
        if not os.path.exists("Results"):
            os.mkdir("Results")
            
        # 视频保存设置
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_filename = f"Results/gameplay_{timestamp}.avi"
        self.VideoWriter = cv2.VideoWriter(self.video_filename, self.fourcc, 50.0,
                                           self.env.observation_space.shape[1::-1])

    def evaluate(self):
        stacked_states = np.zeros((84, 84, 4), dtype=np.uint8)
        mean_ep_reward = []
        obs, int_rewards = [], []
        all_frames = []  # 用于保存所有帧，制作GIF
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for ep in range(self.max_episode):
            episode_frames = []  # 当前episode的帧
            print(f"\nStarting episode {ep + 1}/{self.max_episode}")
            
            # Use gymnasium's reset with seed
            s, _ = self.env.reset(seed=ep)
            stacked_states = stack_states(stacked_states, s, True)
            episode_reward = 0
            clipped_ep_reward = 0
            done = False
            step_count = 0
            
            while not done:
                action, *_ = self.agent.get_actions_and_values(stacked_states)
                # 确保动作是标量整数而非数组
                action_scalar = action.item() if isinstance(action, np.ndarray) else int(action)
                s_, r, done, info = self.env.step(action_scalar)
                episode_reward += r
                clipped_ep_reward += np.sign(r)
                step_count += 1

                stacked_states = stack_states(stacked_states, s_, False)

                int_reward = self.agent.calculate_int_rewards(stacked_states[-1, ...].reshape(1, 84, 84), batch=False)
                int_rewards.append(int_reward)
                obs.append(s_)

                # 保存视频帧
                self.VideoWriter.write(cv2.cvtColor(s_, cv2.COLOR_RGB2BGR))
                
                # 如果需要保存GIF，记录帧数据
                if self.save_gif:
                    # 每隔几帧保存一次，减少GIF大小
                    if step_count % 2 == 0:  # 每2帧保存一次
                        episode_frames.append(s_.copy())
                
                # 渲染显示（如果不是保存GIF模式）
                if not self.save_gif:
                    # 在render_mode='human'模式下，环境会自动渲染，无需额外调用
                    pass
                    
                time.sleep(0.01)
                
            print(f"Episode {ep + 1} completed:")
            print(f"  Steps: {step_count}")
            print(f"  Episode reward: {episode_reward}")
            print(f"  Clipped episode reward: {clipped_ep_reward}")
            if 'episode' in info and 'visited_room' in info['episode']:
                visited_rooms = info['episode']['visited_room']
                print(f"  Visited rooms: {visited_rooms} (Total: {len(visited_rooms)})")
            
            mean_ep_reward.append(episode_reward)
            
            # 保存当前episode的GIF
            if self.save_gif and episode_frames:
                gif_filename = f"Results/episode_{ep+1}_{timestamp}.gif"
                print(f"  Saving episode GIF: {gif_filename}")
                self.save_episode_gif(episode_frames, gif_filename)
                all_frames.extend(episode_frames)
        
        # 保存完整游戏的GIF
        if self.save_gif and all_frames:
            full_gif_filename = f"Results/full_gameplay_{timestamp}.gif"
            print(f"\nSaving complete gameplay GIF: {full_gif_filename}")
            self.save_episode_gif(all_frames, full_gif_filename)
        
        self.env.close()
        self.VideoWriter.release()
        cv2.destroyAllWindows()
        
        print(f"\nEvaluation completed!")
        print(f"Mean episode reward: {sum(mean_ep_reward) / len(mean_ep_reward):0.3f}")
        print(f"Video saved as: {self.video_filename}")
        
        if self.save_gif:
            print(f"GIF files saved in Results/ directory")

    def save_episode_gif(self, frames, filename):
        """保存帧序列为GIF文件"""
        try:
            # 调整帧大小以减少文件大小（可选）
            resized_frames = []
            target_size = (160, 210)  # 原始大小的一半
            
            for frame in frames:
                if frame.shape[:2] != target_size[::-1]:  # 如果需要调整大小
                    resized_frame = cv2.resize(frame, target_size)
                    resized_frames.append(resized_frame)
                else:
                    resized_frames.append(frame)
            
            # 使用imageio保存GIF
            duration = 1.0 / self.gif_fps  # 每帧持续时间
            imageio.mimsave(filename, resized_frames, duration=duration, loop=0)
            print(f"    GIF saved: {filename} ({len(frames)} frames)")
            
        except Exception as e:
            print(f"    Error saving GIF {filename}: {e}")
            # 备用方案：使用PIL
            try:
                pil_frames = [Image.fromarray(frame) for frame in frames[::2]]  # 每隔一帧
                if pil_frames:
                    pil_frames[0].save(
                        filename,
                        save_all=True,
                        append_images=pil_frames[1:],
                        duration=int(1000 / self.gif_fps * 2),  # 毫秒，因为跳帧了所以*2
                        loop=0
                    )
                    print(f"    GIF saved using PIL: {filename}")
            except Exception as e2:
                print(f"    Failed to save GIF with both methods: {e2}")

        # plt.style.use('ggplot')
        # fig = plt.figure()
        # xdata, ydata = [], []
        # plt.subplot(212)
        # ln, = plt.plot([], [])
        #
        # def init():
        #     plt.xlim(0, len(int_rewards))
        #     plt.ylim(0, max(int_rewards))
        #     return ln,
        #
        # def update(frame):
        #     xdata.append(frame)
        #     ydata.append(int_rewards[frame])
        #     ln.set_data(xdata, ydata)
        #     plt.subplot(211)
        #     plt.axis("off")
        #     im = plt.imshow(obs[frame], animated=True)
        #     return ln, im
        #
        # anim = animation.FuncAnimation(fig, update, frames=np.arange(0, len(int_rewards)), init_func=init, interval=3)
        # anim.save('animation.avi', fps=30)
        # # plt.show()
