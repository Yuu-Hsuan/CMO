import os
import logging
import yaml
import argparse
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)

from sample_agent import MyAgent
from pycmo.configs.config import get_config
from pycmo.env.cmo_env import CMOEnv
from pycmo.lib.protocol import SteamClientProps
from pycmo.lib.run_loop import run_loop_steam

def load_config(config_path: str) -> Dict[str, Any]:
    """
    從 YAML 文件讀取配置。
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def update_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    使用命令行參數更新配置。
    """
    if args.scenario_name:
        config['scenario_name'] = args.scenario_name
    if args.player_side:
        config['player_side'] = args.player_side
    if args.scenario_script_folder_name:
        config['scenario_script_folder_name'] = args.scenario_script_folder_name
    if args.ac_name:
        config['agent']['ac_name'] = args.ac_name
    if args.target_name:
        config['agent']['target_name'] = args.target_name
    if hasattr(args, 'destination_lon') and hasattr(args, 'destination_lat'):
        config['agent']['destination'] = {
            'Lon': args.destination_lon,
            'Lat': args.destination_lat
        }
    if args.debug_mode:
        config['debug_mode'] = args.debug_mode
    return config

def main():
    # 設置命令行參數解析
    parser = argparse.ArgumentParser(description="Run CMO agent with configurable parameters.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML config file')
    parser.add_argument('--scenario-name', type=str, help='Name of the scenario')
    parser.add_argument('--player-side', type=str, help='Player side (e.g., Taiwan)')
    parser.add_argument('--scenario-script-folder-name', type=str, help='Folder name containing agent Lua script')
    parser.add_argument('--ac-name', type=str, help='Name of the agent-controlled unit')
    parser.add_argument('--target-name', type=str, help='Name of the target unit')
    parser.add_argument('--debug-mode', type=bool, help='Debug mode')
    args = parser.parse_args()

    # 讀取基礎配置（從 pycmo 自帶的 config）
    base_config = get_config()

    # 讀取 YAML 配置
    yaml_config = load_config(args.config)

    # 使用命令行參數更新配置
    yaml_config = update_config_with_args(yaml_config, args)

    # 提取參數
    scenario_name = yaml_config['scenario_name']
    player_side = yaml_config['player_side']
    scenario_script_folder_name = yaml_config['scenario_script_folder_name']
    ac_name = yaml_config['agent']['ac_name']
    target_name = yaml_config['agent']['target_name']
    debug_mode = yaml_config['debug_mode']

    # 設置文件路徑
    command_version = base_config["command_mo_version"]
    observation_path = os.path.join(base_config['steam_observation_folder_path'], f'{scenario_name}.inst')
    action_path = os.path.join(base_config['steam_observation_folder_path'], "agent_action.lua")
    scen_ended_path = os.path.join(base_config['steam_observation_folder_path'], f'{scenario_name}_scen_has_ended.inst')

    # 初始化 Steam 客戶端屬性
    steam_client_props = SteamClientProps(
        scenario_name=scenario_name,
        agent_action_filename=action_path,
        command_version=command_version
    )

    # 初始化環境
    env = CMOEnv(
        player_side=player_side,
        steam_client_props=steam_client_props,
        observation_path=observation_path,
        action_path=action_path,
        scen_ended_path=scen_ended_path,
    )

    # 初始化 Agent
    dest = yaml_config['agent'].get('destination', None)
    agent = MyAgent(
        player_side=player_side,
        ac_name=ac_name,
        target_name=target_name,     # 敵艦用來偵測的 target_name
        destination=dest             # 金門座標
    )

    # 執行主循環
    run_loop_steam(env=env, agent=agent, max_steps=None)

if __name__ == "__main__":
    main()
