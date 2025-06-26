#!/usr/bin/env python3
"""
Simple test script to verify that the environment setup works correctly.
"""

import torch
import numpy as np
import gymnasium as gym

# Import and register Atari environments
try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    print("Warning: Could not import ale_py")

from Common.config import get_params
from Common.utils import make_atari
from Brain.brain import Brain

def test_environment():
    """Test if the environment can be created and basic operations work."""
    print("Testing environment setup...")
    
    # Test configuration
    config = get_params()
    print(f"Config loaded successfully: {config['env_name']}")
    
    # Test environment creation
    try:
        env = make_atari(config["env_name"], config["max_frames_per_episode"])
        print("✓ Environment created successfully")
        
        # Test environment step
        state, _ = env.reset()
        print(f"✓ Environment reset: state shape {state.shape}")
        
        action = env.action_space.sample()
        print(f"Testing env.step with action: {action}")
        
        # Our wrapper returns 4 values (state, reward, done, info)
        next_state, reward, done, info = env.step(action)
        print(f"✓ Environment step: reward={reward}, done={done}")
        
        env.close()
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test Brain initialization
    try:
        # Use direct gymnasium call for getting action space
        test_env = gym.make(config["env_name"])
        config.update({"n_actions": test_env.action_space.n})
        test_env.close()
        
        config.update({"batch_size": (config["rollout_length"] * config["n_workers"]) // config["n_mini_batch"]})
        config.update({"predictor_proportion": 32 / config["n_workers"]})
        
        brain = Brain(**config)
        print("✓ Brain initialized successfully")
        
        # Test tensor types
        test_state = np.random.randint(0, 255, config["state_shape"], dtype=np.uint8)
        actions, int_values, ext_values, log_probs, action_probs = brain.get_actions_and_values(test_state)
        print("✓ Brain inference working")
        
    except Exception as e:
        print(f"✗ Brain test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✓ All tests passed!")
    return True

if __name__ == "__main__":
    # Set minimal arguments for testing
    import sys
    sys.argv = ['test_environment.py', '--n_workers', '2', '--do_test']
    
    success = test_environment()
    if success:
        print("\n🎉 Environment is ready for training!")
    else:
        print("\n❌ Please check the error messages above")
