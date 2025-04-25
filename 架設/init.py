### 將 C:\Users\yuhsu\OneDrive\桌面\pycmopen\pycmo\configs\config.py 內容改為:
import os

def get_config():
    pycmo_path = os.path.join("C:/Users/yuhsu/OneDrive/桌面", "pycmopen")
    cmo_path = os.path.join("C:/Program Files (x86)/Steam/steamapps/common", "Command - Modern Operations")
    command_mo_version = "Command: Modern Operations v1.07 - Build 1567.6"
    use_gymnasium = False

    return {
    "command_path": cmo_path,
    "pycmo_path": pycmo_path,
    "observation_path": os.path.join(pycmo_path, "xml", "steps"),
    "steam_observation_folder_path": os.path.join(cmo_path, 'ImportExport'),
    "scen_ended": os.path.join(pycmo_path, "pycmo", "configs", "scen_has_ended.txt"),
    "pickle_path": os.path.join(pycmo_path, "pickle"),
    "scripts_path": os.path.join(pycmo_path, "scripts"),
    "command_mo_version": command_mo_version,
    "gymnasium": use_gymnasium,
    # "command_cli_output_path": "C:\\ProgramData\\Command Professional Edition 2\\Analysis_Int", # only applicable to Premium version so we update this later
    }
