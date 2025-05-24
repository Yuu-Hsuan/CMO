# Author: Minh Hua
# Date: 08/16/2021
# Last Update: 06/16/2022
# Purpose: Export Command raw data into Features object and flatten into agent-consummable data.

# imports
import xml.etree.ElementTree as ET
import xmltodict
from typing import Tuple, NamedTuple
import logging

# This section can be modified to dictate the type of observations that are returned from the game at each time step
# Game
class Game(NamedTuple):
    TimelineID: str | None
    Time: int
    ScenarioName: str
    ZeroHour: int
    StartTime: int
    Duration: int
    Sides: list[str]

# Side 
class Side(NamedTuple):
    ID: str
    Name: str
    TotalScore: int

# Weapon
class Weapon(NamedTuple):
    XML_ID: int
    ID: str
    WeaponID: int
    QuantRemaining: int | None
    MaxQuant: int | None

# Contacts 
class Contact(NamedTuple):
    XML_ID: int
    ID: str
    Name: str | None
    CS: float | None 
    CA: float | None 
    Lon: float | None 
    Lat: float | None

# Mount 
class Mount(NamedTuple):
    XML_ID: int
    ID: str
    Name: str
    DBID: int
    Weapons: list[Weapon]

# Loadout
class Loadout(NamedTuple):
    XML_ID: int
    ID: int
    Name: str
    DBID: int
    Weapons: list[Weapon]

# Unit 
class Unit(NamedTuple):
    XML_ID: int
    ID: str
    Name: str
    Side: str
    DBID: int
    Type: str
    CH: float | None
    CS: float | None
    CA: float | None
    Lon: float
    Lat: float
    CurrentFuel: float | None
    MaxFuel: float | None
    Mounts: list[Mount] | None
    Loadout: Loadout | None

class Mission(NamedTuple):
    ID: str
    Name: str
    Side: str
    Type: str
    Subtype: str
    IsActive: bool
    StartTime: str
    EndTime: str
    SISH: bool
    AAR: list
    UnitList: list

class Features(object):
    """
    Render feature layers from a Command: Professional Edition scenario XML into named tuples.
    """
    def __init__(self, xml:str, player_side:str) -> None:
        """
        Description:
            Initialize a Features object to hold observations.

        Keyword Arguments:
            xml: the path to the xml file containing the game observations.
            player_side: the side of the player. Dictates the units that they can actually control.
        
        Returns:
            None
        """
        try:
            tree = ET.parse(xml) # This variable contains the XML tree
            root = tree.getroot() # This is the root of the XML tree
            xmlstr = ET.tostring(root)            
            self.scen_dic = xmltodict.parse(xmlstr) # our scenario xml is now in 'dic'
        except FileNotFoundError:
            raise FileNotFoundError("Unable to parse scenario xml.")
        
        self.logger = logging.getLogger(__name__)
        
        # get features
        self.player_side = player_side
        self.meta = self.get_meta() # Return the scenario-level rmation of the scenario
        self.units = self.get_side_units(player_side)
        try:
            player_side_index = self.get_sides().index(player_side)
        except ValueError:
            raise ValueError('Cannot find player side.')
        self.side_ = self.get_side_properties(player_side_index)
        self.contacts = self.get_side_contacts(player_side_index)
        self.missions = self.get_missions(player_side_index)

    def transform_obs_into_arrays(self) -> Tuple[Game, list[Unit], Side, list[Contact]]:
        """
        Description:
            Render some Command observations into something an agent can handle.

        Keyword Arguments:
            None
        
        Returns:
            (list) a list of the game's meta data, units, side info, and contact info
        """
        observation = (self.meta, self.units, self.side_, self.contacts, self.missions)
        return observation

    # ============== XML Data Extraction Methods ================================
    def get_meta(self) -> Game:
        """
        Description:
            Get meta data (top-level scenario data).

        Keyword Arguments:
            None
        
        Returns:
            (Game) a named tuple containing the meta data of the game.
        """
        try:
            timeline_id = self.scen_dic['Scenario']["TimelineID"]
        except KeyError:
            timeline_id = None
            
        try:
            return Game(timeline_id,
                        int(self.scen_dic['Scenario']["Time"]),
                        self.scen_dic['Scenario']["Title"],
                        int(self.scen_dic['Scenario']["ZeroHour"]),
                        int(self.scen_dic['Scenario']["StartTime"]),
                        int(self.scen_dic['Scenario']["Duration"]),
                        self.get_sides())
        except KeyError:
            raise KeyError("Failed to get scenario properties.")

    def get_sides(self) -> list[str]:
        """
        Description:
            Get the number and names of all sides in a scenario.

        Keyword Arguments:
            None
        
        Returns:
            (list) a list of side names
        """        
        try:
            return [side['Name'] for side in self.scen_dic['Scenario']['Sides']['Side']]
        except KeyError:
            raise KeyError("Failed to get list of side names in scenario.")

    def get_side_properties(self, side_index:int=0) -> Side:
        """
        Description:
            Get the properties (score, name, etc.) of a side.

        Keyword Arguments:
            side_index: index to choose the side for information.
        
        Returns:
            (Side) a named tuple containing Side information.
        """
        try:
            side = self.scen_dic['Scenario']['Sides']['Side'][side_index]
            return Side(side['ID'], side['Name'], int(side['TotalScore']))
        except KeyError:
            raise KeyError("Failed to get side properties.")

    def get_side_units(self, side_name=None) -> list[Unit]:
        """
        Description:
            Get all the units of a side.

        Keyword Arguments:
            side_name: the name of the side to get units for.
        
        Returns:
            (list) a list of the units of the side.
        """
        unit_ids = []
        if side_name == None or 'ActiveUnits' not in self.scen_dic["Scenario"].keys():
            return unit_ids
        for unit_type in self.scen_dic["Scenario"]["ActiveUnits"].keys():
            active_units = self.scen_dic["Scenario"]["ActiveUnits"][unit_type]
            if not isinstance(self.scen_dic["Scenario"]["ActiveUnits"][unit_type], list):
                active_units = [self.scen_dic["Scenario"]["ActiveUnits"][unit_type]]
            for unit_idx, unit in enumerate(active_units):
                try:
                    if unit["Side"] == side_name:
                        unit_ids.append(self.get_unit(unit=unit, unit_idx=unit_idx, unit_type=unit_type, side_name=side_name))
                except KeyError:
                    self.logger.warn("Failed to parse one unit xml.")                 
        return unit_ids
    
    def get_unit(self, unit:dict, unit_idx:int, unit_type:str, side_name:str) -> Unit:
        try:
            unit_id = unit['ID']
            name = unit['Name']
            dbid = int(unit['DBID'])
            lon = float(unit['Lon'])
            lat = float(unit['Lat'])
            ch = None
            cs = None
            ca = None
            loadout = None
            mount = None
            cf = None
            mf = None
            # print("unit:", unit)
            # print("unit['CS']", unit['CS'])
            if 'Loadout' in unit.keys() and unit['Loadout'] != None:
                loadout = self.get_loadout(unit)
            if 'Mounts' in unit.keys() and unit['Mounts'] != None:
                mount = self.get_mount(unit)
            if 'CH' in unit.keys() and unit['CH'] != None:
                ch = float(unit['CH'])
            if 'CS' in unit.keys() and unit['CS'] != None:
                cs = float(unit['CS'])
            if 'CA' in unit.keys() and unit['CA'] != None:
                ca = float(unit['CA'])
            # 確保 Fuel 存在
            if 'Fuel' in unit:
                fuel_rec = unit['Fuel'].get('FuelRec', None)
                # 如果 FuelRec 是 list，取第一個元素
                if isinstance(fuel_rec, list):
                    fuel_rec = fuel_rec[0] if fuel_rec else {}
                # 確保 FuelRec 存在並取得 CQ 和 MQ
                if isinstance(fuel_rec, dict):
                    cf = float(fuel_rec.get('CQ', 0))  # 預設為 0
                    mf = float(fuel_rec.get('MQ', 0))  # 預設為 0

            return Unit(unit_idx, unit_id, name, side_name, dbid, unit_type, ch, cs, ca, lon, lat, cf, mf, mount, loadout)
        except KeyError:
            raise KeyError("Error parsing xml for unit.")

    def get_mount(self, unit:dict) -> list[Mount]:
        """
        Description:
            Returns the Mounts of a unit. Required to control the weapons on the unit.

        Keyword Arguments:
            unit: the xml string of the unit. Preferably in dictionary format.
        
        Returns:
            (list) a list of the unit's mounts.
        """        
        parsed_mounts = []
        mounts = unit["Mounts"]["Mount"]
        if not isinstance(mounts, list):
            mounts = [unit["Mounts"]["Mount"]]
        for mount_idx, mount in enumerate(mounts):
            mount_id = mount["ID"]
            name = mount["Name"]
            dbid = int(mount["DBID"])
            parsed_mounts.append(Mount(mount_idx, mount_id, name, dbid, self.get_loadout_or_mount_weapons('Mount', mount)))
        return parsed_mounts      

    def get_loadout(self, unit:dict) -> Loadout:
        """
        Description:
            Returns the Loadout of a unit.

        Keyword Arguments:
            unit: the xml string of the unit. Preferably in dictionary format.
        
        Returns:
            (Loadout) the unit's current loadout.
        """                
        loadout = unit["Loadout"]["Loadout"]
        loadout_id = int(loadout["ID"])
        name = loadout["Name"]
        dbid = int(loadout["DBID"])
        return Loadout(0, loadout_id, name, dbid, self.get_loadout_or_mount_weapons('Loadout', loadout))
    
    def get_loadout_or_mount_weapons(self, mount_or_loadout:str, xml_str:dict) -> list[Weapon]:
        """
        Description:
            Returns the weapons on a mount or loadout. Also sets the unit's list of available weapons.

        Keyword Arguments:
            mount_or_loadout: str to determine whether we are getting weapons from a mount or a loadout.
            xml_str: the xml str of the unit.
        
        Returns:
            (list) the unit's weapons.
        """
        weapons = []

        if mount_or_loadout == "Loadout":
            if 'Weaps' not in xml_str.keys() or xml_str['Weaps'] == None:
                return weapons
            wrec = xml_str['Weaps']['WRec']
        elif mount_or_loadout == "Mount":
            if 'MW' not in xml_str.keys() or xml_str['MW'] == None:
                return weapons
            wrec = xml_str['MW']['WRec']
        else:
            return weapons
        
        return self.get_weapon_records(weapon_records=wrec)
    
    def get_weapon_records(self, weapon_records:dict) -> list[Weapon]:
        weapons = []

        if not isinstance(weapon_records, list):
            weapon_records = [weapon_records]
        
        for weapon_record_idx, weapon_record in enumerate(weapon_records):
            cl = None
            ml = None
            if "CL" in weapon_record.keys():
                cl = int(weapon_record["CL"])
            if "ML" in weapon_record.keys():
                ml = int(weapon_record["ML"])
            weapon = Weapon(weapon_record_idx, weapon_record["ID"], int(weapon_record['WeapID']), cl, ml)     
            weapons.append(weapon)   

        return weapons   

    def get_side_contacts(self, side_index):
        # 確保 scen_dic 存在
        if self.scen_dic is None:
            print(" 錯誤: `scen_dic` 為 None，CMO 可能沒有正確輸出數據。")
            return []
        # 確保 Scenario, Sides, Side 存在
        if "Scenario" not in self.scen_dic or "Sides" not in self.scen_dic["Scenario"]:
            print(" 錯誤: `scen_dic` 缺少 'Scenario' 或 'Sides'，請確認 CMO 是否正確輸出數據。")
            return []
        if "Side" not in self.scen_dic["Scenario"]["Sides"]:
            print(" CMO 沒有偵測到任何側邊 (Side)。")
            return []
        sides = self.scen_dic["Scenario"]["Sides"]["Side"]
        # 確保 side_index 不超過範圍
        if side_index >= len(sides):
            print(f" 錯誤: `side_index` ({side_index}) 超過可用範圍 ({len(sides)})。")
            return []
        if "Contacts" not in sides[side_index] or sides[side_index]["Contacts"] is None:
            print(f" `{sides[side_index]['Name']}` 沒有任何 `Contacts`。")
            return []
        contacts = sides[side_index]["Contacts"].get("Contact", [])
        # 確保 contacts 是 list
        if not isinstance(contacts, list):
            contacts = [contacts]
        return contacts

    
    def get_contact(self, contact:dict, contact_idx:int) -> Contact:
        try:
            cs = None
            ca = None
            lon = None
            lat = None
            contact_name = None
            if 'CS' in contact.keys() and contact['CS'] != None:
                cs = float(contact['CS'])
            if 'CA' in contact.keys() and contact['CA'] != None:
                ca = float(contact['CA'])
            if 'Lon' in contact.keys() and contact['Lon'] != None:
                lon = float(contact['Lon'])
            if 'Lat' in contact.keys() and contact['Lat'] != None:
                lat = float(contact['Lat'])
            if 'Name' in contact.keys() and contact['Name'] != None:
                contact_name = contact['Name']
            return Contact(contact_idx, contact["ID"], contact_name, cs, ca, lon, lat)
        except KeyError:
            raise KeyError("Error parsing xml for contact.")

    def get_missions(self, side_index:int=0) -> list[Mission]:
        """
        Description:
            Return the missions of a side.
            If no missions exist, return an empty list.
        """
        try:
            missions = self.scen_dic["Scenario"]["Sides"]["Side"][side_index]["Missions"].get("Mission", [])
            if not isinstance(missions, list):
                missions = [missions]
            return missions
        except (KeyError, TypeError, AttributeError):
            # 如果 Missions 不存在或為空，返回空列表
            return []

class FeaturesFromSteam(Features):
    """
    Renders feature layers from a Command: Modern Operations scenario XML into named tuples.
    """
    def __init__(self, xml:str, player_side:str) -> None:
        """
        Description:
            Initialize a Features object to hold observations.

        Keyword Arguments:
            xml: the path to the xml file containing the game observations.
            player_side: the side of the player. Dictates the units that they can actually control.
        
        Returns:
            None
        """
        try:         
            self.scen_dic = xmltodict.parse(xml) # our scenario xml is now in 'dic'
        except FileNotFoundError:
            raise FileNotFoundError("Unable to parse scenario xml.")
        
        # get features
        self.player_side = player_side
        self.meta = self.get_meta() # Return the scenario-level rmation of the scenario
        self.units = self.get_side_units(player_side)
        try:
            player_side_index = self.get_sides().index(player_side)
        except ValueError:
            raise ValueError('Cannot find player side.')
        self.side_ = self.get_side_properties(player_side_index)
        self.contacts = self.get_side_contacts(player_side_index)
        self.missions = self.get_missions(player_side_index)
