#!/usr/bin/env python3
"""
Check available Atari environments
"""

import gymnasium as gym
# Import Atari environments to register them
try:
    import gymnasium.envs.atari
except ImportError:
    print("Warning: Could not import gymnasium.envs.atari")

try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    print("Warning: Could not import ale_py")

def check_atari_envs():
    """Check available Atari environments"""
    print("Checking available Atari environments...")
    
    # Get all registered environments
    all_envs = gym.envs.registry.keys()
    
    # Filter for Montezuma environments
    montezuma_envs = [env for env in all_envs if 'Montezuma' in env]
    
    print(f"Found {len(montezuma_envs)} Montezuma environments:")
    for env in sorted(montezuma_envs):
        print(f"  - {env}")
    
    print("\nAll Atari environments:")
    atari_envs = [env for env in all_envs if 'ALE/' in env or any(game in env for game in ['Breakout', 'Pong', 'SpaceInvaders', 'Montezuma', 'Asterix', 'Freeway'])]
    for env in sorted(atari_envs)[:20]:  # Show first 20
        print(f"  - {env}")
    
    if len(atari_envs) > 20:
        print(f"  ... and {len(atari_envs) - 20} more")
    
    # Try to create the environment with different names
    possible_names = [
        'MontezumaRevengeNoFrameskip-v4',
        'ALE/MontezumaRevenge-v5',
        'MontezumaRevenge-v4',
        'MontezumaRevenge-v5'
    ]
    
    print(f"\nTesting possible environment names:")
    for env_name in possible_names:
        try:
            env = gym.make(env_name)
            print(f"✓ {env_name} - SUCCESS")
            env.close()
            return env_name
        except Exception as e:
            print(f"✗ {env_name} - FAILED: {str(e)[:100]}")
    
    return None

if __name__ == "__main__":
    working_env = check_atari_envs()
    if working_env:
        print(f"\n✅ Recommended environment name: {working_env}")
    else:
        print(f"\n❌ No working Montezuma environment found. You may need to install the Atari ROMs.")
        print("Try running: uv run AutoROM --accept-license")
