# Author: Minh Hua
# Date: 08/16/2021
# Last Update: 11/24/2023
# Purpose: Encodes the action space of the game.

# imports
import collections

from typing import Tuple
from random import uniform, randint

from pycmo.lib.features import Features, FeaturesFromSteam, Unit

Function = collections.namedtuple("Function", ['id', 'name', 'corresponding_def', 'args', 'arg_types'])

# canned actions that can be called by an agent to send an action
def no_op():
  return ""

def launch_aircraft(side:str, unit_name:str, launch:bool=True) -> str:
  return f"ScenEdit_SetUnit({{side = '{side}', name = '{unit_name}', Launch = {'true' if launch else 'false'}}})"

def set_unit_course(side:str, unit_name:str, latitude:float, longitude:float) -> str:
  return f"ScenEdit_SetUnit({{side = '{side}', name = '{unit_name}', course = {{{{longitude = {longitude}, latitude = {latitude}, TypeOf = 'ManualPlottedCourseWaypoint'}}}}}})"

def manual_attack_contact(attacker_id:str, contact_id:str, weapon_id:int, qty:int, mount_id:int=None) -> str:
  return f"ScenEdit_AttackContact('{attacker_id}', '{contact_id}' , {{mode='1', " + (f"mount='{mount_id}', " if mount_id else "") + f"weapon='{weapon_id}', qty='{qty}'}})"

def auto_attack_contact(attacker_id:str, contact_id:str) -> str:
  return f"ScenEdit_AttackContact('{attacker_id}', '{contact_id}', {{mode='0'}})"

def refuel_unit(side:str, unit_name:str, tanker_name:str) -> str:
  return f"ScenEdit_RefuelUnit({{side='{side}', unitname='{unit_name}', tanker='{tanker_name}'}})"

def auto_refuel_unit(side:str, unit_name:str) -> str:
  return f"ScenEdit_RefuelUnit({{side='{side}', unitname='{unit_name}'}})"

def rtb(side:str, unit_name:str, return_to_base:bool=True) -> str:
  return f"ScenEdit_SetUnit({{side = '{side}', name = '{unit_name}', RTB = {'true' if return_to_base else 'false'}}})"

### NEW ACTIONS


def set_unit_position(side:str, unit_name:str, latitude:float, longitude:float) -> str:
  return f"ScenEdit_SetUnit({{side = '{side}', name = '{unit_name}', latitude = {latitude}, longitude = {longitude}}})"
def set_unit_heading_and_speed(side:str, unit_name:str, heading:float, speed:float) -> str:
  return f"ScenEdit_SetUnit({{side = '{side}', name = '{unit_name}', desiredHeading = {heading}, Speed = {speed}}})"


def set_you_me_position(my_side:str, my_unit_name:str, my_latitude:float, my_longitude:float, your_side:str, your_unit_name:str, your_latitude:float, your_longitude:float) -> str:
  return f"ScenEdit_SetUnit({{side = '{my_side}', name = '{my_unit_name}', latitude = {my_latitude}, longitude = {my_longitude}}})\nScenEdit_SetUnit({{side = '{your_side}', name = '{your_unit_name}', latitude = {your_latitude}, longitude = {your_longitude}}})"

def set_you_me_side(my_side:str, my_unit_name:str, my_heading:float, my_speed:float, your_side:str, your_unit_name:str, your_heading:float, your_speed:float) -> str:
  return f"ScenEdit_SetUnit({{side = '{my_side}', name = '{my_unit_name}', desiredHeading = {my_heading}, Speed = {my_speed}}})\nScenEdit_SetUnit({{side = '{your_side}', name = '{your_unit_name}', heading = {your_heading}, speed = {your_speed}}})"


def assign_mission(your_unit_name: str) -> str:
  return f'ScenEdit_AssignUnitToMission("{your_unit_name}", "Kinmen patrol")'

def reset_my_side (type: str, my_unit_name: str, my_side: str, my_dbid: int, my_latitude: float, my_longitude: float) -> str:
  return f"ScenEdit_AddUnit({{type = '{type}', name = '{my_unit_name}', side = '{my_side}', dbid = {my_dbid}, latitude = {my_latitude}, longitude = {my_longitude}}})"

def reset_scen_position(
    type: str, 
    my_unit_name: str, 
    my_side: str, my_dbid: int, 
    my_latitude: float, my_longitude: float,
    your_unit_name: str, 
    your_side: str, your_dbid: int,
    your_latitude: float, your_longitude: float) -> str:
    return (
        f"ScenEdit_DeleteUnit({{side = '{my_side}', name = '{my_unit_name}'}})\n"
        f"ScenEdit_AddUnit({{type = '{type}', name = '{my_unit_name}', side = '{my_side}', dbid = {my_dbid}, latitude = {my_latitude}, longitude = {my_longitude}}})\n"
        f"ScenEdit_DeleteUnit({{side = '{your_side}', name = '{your_unit_name}'}})\n"
        f"ScenEdit_AddUnit({{type = '{type}', name = '{your_unit_name}', side = '{your_side}', dbid = {your_dbid}, latitude = {your_latitude}, longitude = {your_longitude}}})\n"
        f'ScenEdit_AssignUnitToMission("{your_unit_name}", "Kinmen patrol")'
        )
# f"ScenEdit_SetUnit({{side = '{my_side}', name = '{my_unit_name}', latitude = {my_latitude}, longitude = {my_longitude}}})\n"


ARG_TYPES = {
  'no_op': ['NoneChoice'],
  'launch_aircraft': ['EnumChoice', 'EnumChoice', 'EnumChoice'],
  'set_unit_course': ['EnumChoice', 'EnumChoice', 'Range', 'Range'],
  'manual_attack_contact': ['EnumChoice', 'EnumChoice', 'EnumChoice', 'EnumChoice', 'EnumChoice'],
  'auto_attack_contact': ['EnumChoice', 'EnumChoice'],
  'refuel_unit': ['EnumChoice', 'EnumChoice', 'EnumChoice'],
  'auto_refuel_unit': ['EnumChoice', 'EnumChoice'],
  'rtb': ['EnumChoice', 'EnumChoice', 'EnumChoice'],
}

class AvailableFunctions():
  def __init__(self, features:Features | FeaturesFromSteam):
    self.ARG_TYPES = ARG_TYPES
    self.refresh(features=features)

  def refresh(self, features:Features|FeaturesFromSteam) -> None:
    self.sides = [features.player_side]

    self.unit_ids, self.unit_names = self.get_unit_ids_and_names(features)
    self.contact_ids = self.get_contact_ids(features)
    self.mount_ids, self.loadout_ids, self.weapon_ids, self.weapon_qtys = self.get_weapons(features)

    self.VALID_FUNCTIONS = self.get_valid_functions()

  def get_unit_ids_and_names(self, features:Features|FeaturesFromSteam) -> Tuple[list[str], list[str]]:
    unit_ids = []
    unit_names = []
    for unit in features.units:
      unit_ids.append(unit.ID)
      unit_names.append(unit.Name)
    return unit_ids, unit_names

  def get_contact_ids(self, features: Features | FeaturesFromSteam) -> list[str]:
    # 若 features 沒有 contacts 屬性或 contacts 為空，則回傳空列表
    if not hasattr(features, "contacts") or not features.contacts:
      return []
    contact_ids = []
    for contact in features.contacts:
      # 如果 contact 為字典，就用字典方式取值；否則用屬性方式
      if isinstance(contact, dict):
        cid = contact.get("ID")
      else:
        cid = getattr(contact, "ID", None)
      # 只有當 cid 存在時才加入列表
      if cid is not None:
        contact_ids.append(cid)
    return contact_ids

  
  def get_weapons(self, features:Features|FeaturesFromSteam) -> Tuple[list[int], list[int], list[int], list[int]]:
    mount_ids = []
    loadout_ids = []
    weapon_ids = []
    weapon_qtys = []
    for unit in features.units:
      unit_mount_ids, unit_mount_weapon_ids, unit_mount_weapon_qtys = self.get_mount_ids_weapon_ids_and_qtys(unit)
      unit_loadout_id, unit_loadout_weapon_ids, unit_loadout_weapon_qtys = self.get_loadout_id_weapon_ids_and_qtys(unit)
      mount_ids += unit_mount_ids
      if unit_loadout_id: loadout_ids.append(unit_loadout_id)
      weapon_ids += unit_mount_weapon_ids + unit_loadout_weapon_ids
      weapon_qtys += unit_mount_weapon_qtys + unit_loadout_weapon_qtys
    return mount_ids, loadout_ids, weapon_ids, weapon_qtys
  
  def get_mount_ids_weapon_ids_and_qtys(self, unit:Unit) -> Tuple[list[int], list[int], list[int]]:
    mount_ids = []
    weapon_ids = []
    weapon_qtys = []
    if unit.Mounts:
      for mount in unit.Mounts:
        mount_ids.append(mount.DBID)
        for weapon in mount.Weapons:
          weapon_ids.append(weapon.WeaponID)
          weapon_qtys.append(weapon.QuantRemaining)
    return mount_ids, weapon_ids, weapon_qtys

  def get_loadout_id_weapon_ids_and_qtys(self, unit:Unit) -> Tuple[int | None, list[int], list[int]]:
    loadout_id = None
    weapon_ids = []
    weapon_qtys = []
    loadout = unit.Loadout
    if loadout:
      loadout_id = loadout.DBID
      for weapon in loadout.Weapons:
        weapon_ids.append(weapon.WeaponID)
        weapon_qtys.append(weapon.QuantRemaining)
    return loadout_id, weapon_ids, weapon_qtys
  
  def get_valid_function_args(self) -> dict[str, list]:
    boolean_list = [True, False]
    latitude_ranges = [-90.0, 90.0]
    longitude_ranges = [-180.0, 180.0]
    return {
      'no_op': [],
      'launch_aircraft': [self.sides, self.unit_names, boolean_list],
      'set_unit_course': [self.sides, self.unit_names, latitude_ranges, longitude_ranges],
      'manual_attack_contact': [self.unit_ids, self.contact_ids, self.weapon_ids, self.weapon_qtys, self.mount_ids],
      'auto_attack_contact': [self.unit_ids, self.contact_ids],
      'refuel_unit': [self.sides, self.unit_names, self.unit_names],
      'auto_refuel_unit': [self.sides, self.unit_names],
      'rtb': [self.sides, self.unit_names, boolean_list]
    }
  
  def get_valid_functions(self) -> list[Function]:
    VALID_FUNCTION_ARGS = self.get_valid_function_args()
    valid_functions = [
      Function(0, "no_op", no_op, VALID_FUNCTION_ARGS['no_op'], ARG_TYPES['no_op']),
      Function(1, "launch_aircraft", launch_aircraft, VALID_FUNCTION_ARGS['launch_aircraft'], ARG_TYPES['launch_aircraft']),
      Function(2, 'set_unit_course', set_unit_course, VALID_FUNCTION_ARGS['set_unit_course'], ARG_TYPES['set_unit_course']),
      Function(3, "manual_attack_contact", manual_attack_contact, VALID_FUNCTION_ARGS['manual_attack_contact'], ARG_TYPES['manual_attack_contact']),
      Function(4, "auto_attack_contact", auto_attack_contact, VALID_FUNCTION_ARGS['auto_attack_contact'], ARG_TYPES['auto_attack_contact']),
      Function(5, 'refuel_unit', refuel_unit, VALID_FUNCTION_ARGS['refuel_unit'], ARG_TYPES['refuel_unit']),
      Function(6, 'auto_refuel_unit', auto_refuel_unit, VALID_FUNCTION_ARGS['auto_refuel_unit'], ARG_TYPES['auto_refuel_unit']),
      Function(7, 'rtb', rtb, VALID_FUNCTION_ARGS['rtb'], ARG_TYPES['rtb'])
    ]
    return valid_functions
  
  def sample(self) -> str:
    random_function = self.VALID_FUNCTIONS[randint(0, len(self.VALID_FUNCTIONS) - 1)]
    if len(random_function.arg_types) == 1 and random_function.arg_types[0] == "NoneChoice":
      return random_function.corresponding_def()

    function_args = []
    for valid_args, arg_type in zip(random_function.args, random_function.arg_types):
      if arg_type == "EnumChoice":
        arg = valid_args[randint(0, len(valid_args) - 1)]
      elif arg_type == "Range":
        arg = uniform(valid_args[0], valid_args[1])
      function_args.append(arg)
    
    return random_function.corresponding_def(*function_args)

  def contains(self, function_id:int, function_args:list) -> bool:
    if function_id < 0 or function_id > len(self.VALID_FUNCTIONS) or (function_id == 0 and len(function_args) > 0):
      return False
    else:
      valid_function_args = self.VALID_FUNCTIONS[function_id].args
      valid_function_arg_types = self.VALID_FUNCTIONS[function_id].arg_types
      if len(function_args) != len(valid_function_args): return False
      for function_arg, valid_args, arg_type in zip(function_args, valid_function_args, valid_function_arg_types):
        if arg_type == "EnumChoice" and function_arg not in valid_args: 
          return False
        elif arg_type == "Range" and (function_arg < valid_args[0] or function_arg > valid_args[1]): 
          return False
      return True