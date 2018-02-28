import math
import numpy
import random
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import units as pu
import queue


class brain:
  screen_width = 84
  screen_height = 84
  minimap_width = 64
  minimap_height = 64

  minimap_x_min = 0
  minimap_x_max = 64
  minimap_y_min = 0
  minimap_y_max = 64
  minimap_trimmed_width = 64
  minimap_trimmed_height = 64

  attack_minimap_width = 64
  attack_minimap_height = 64

  def __init__(self,
               mirror_map=False,
               state=None,
               attack_minimap_width=64,
               attack_minimap_height=64):
    if state is None:
      state = state_manager(mirror_map)
    self.state = state
    self.mirror_map = mirror_map
    self.attack_minimap_width = attack_minimap_width
    self.attack_minimap_height = attack_minimap_height
    self.reset()

  def observe(self, obs):
    self.obs = obs
    if obs.first():
      player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                            features.PlayerRelative.SELF).nonzero()
      self.base_top_left = (player_y.any() and player_y.mean() <= 31)

      # Trim non-terrain off the minimap
      height_y, height_x = obs.observation.feature_minimap.height_map.nonzero()
      self.minimap_y_min = height_y.min()
      self.minimap_y_max = height_y.max()
      self.minimap_x_min = height_x.min()
      self.minimap_x_max = height_x.max()
      self.minimap_trimmed_width = self.minimap_x_max - self.minimap_x_min
      self.minimap_trimmed_height = self.minimap_y_max - self.minimap_y_min
    self.state.observe(obs)

  def reset(self):
    self.action_queue = queue.Queue()
    self.researched_combat_shield = False
    self.base_top_left = None
    self.infantry_armor_level = 0
    self.infantry_weapons_level = 0

  def clean_screen_location(self, x, y):
    if x < 0:
      x = 0
    if x > (self.screen_width - 1):
      x = (self.screen_width - 1)
    if y < 0:
      y = 0
    if y > (self.screen_height - 1):
      y = (self.screen_height - 1)
    return [x, y]

  def clean_minimap_location(self, x, y):
    if x < 0:
      x = 0
    if x > (self.minimap_width - 1):
      x = (self.minimap_width - 1)
    if y < 0:
      y = 0
    if y > (self.minimap_height - 1):
      y = (self.minimap_height - 1)
    return [x, y]

  def queue_action(self, action, params=[]):
    self.action_queue.put((action, params))

  def has_queued_action(self):
    return self.action_queue.qsize() > 0

  def get_queued_action(self):
    action, params = self.action_queue.get()
    result = getattr(self, action)(*params)
    if result is not None:
      return result
    self.action_queue = queue.Queue()
    return actions.FUNCTIONS.no_op()

  def mirror_location(self, x, y):
    if self.mirror_map and not self.base_top_left:
      return (self.minimap_x_max - x, self.minimap_y_max - y)
    return (self.minimap_x_min + x, self.minimap_y_min + y)

  def action_available(self, action):
    return action in self.obs.observation.available_actions

  def get_units_by_types(self, unit_types):
    units = [unit for unit in self.obs.observation.feature_units
             if unit.unit_type in unit_types]
    return units

  def has_unit_at_location(self, units, x, y, margin):
    for unit in units:
      if unit.x > x - margin  and unit.x < x + margin \
          and unit.y > y - margin and unit.y < y + margin:
        return True
    return False

  def get_unit_by_type_at_location(self, unit_types, x, y, margin):
    units = self.get_units_by_types(unit_types)
    for unit in units:
      if unit.x > x - margin  and unit.x < x + margin \
          and unit.y > y - margin and unit.y < y + margin:
        return unit
    return None

  def has_unit_type_at_location(self, unit_types, x, y, margin=5):
    units = self.get_units_by_types(unit_types)
    return self.has_unit_at_location(units, x, y, margin)

  def get_random_unit(self, units):
    i = random.randint(0, len(units) - 1)
    return units[i]

  def select_random_unit(self, unit_types, select_type="select"):
    units = self.get_units_by_types(unit_types)
    if len(units) > 0:
      unit = self.get_random_unit(units)
      return self.select_point(self.clean_screen_location(unit.x, unit.y),
                               select_type)

  def select_point(self, location, select_type="select"):
    return actions.FUNCTIONS.select_point(select_type, location)

  def has_barracks_at_location(self, x, y, margin=5):
    return self.has_unit_type_at_location(
        [pu.Terran.Barracks, pu.Terran.BarracksFlying], x, y, margin)

  def has_reactor_at_location(self, x, y, margin=5):
    return self.has_unit_type_at_location(
        [pu.Terran.Reactor, pu.Terran.BarracksReactor], x, y, margin)

  def get_barracks_at_location(self, x, y, margin=5):
    return self.get_unit_by_type_at_location(
        [pu.Terran.Barracks, pu.Terran.BarracksFlying], x, y, margin)

  def select_worker(self, idle_only=False):
    if self.obs.observation.player.idle_worker_count > 0:
      return actions.FUNCTIONS.select_idle_worker("select")
    if not idle_only:
      return self.select_random_unit([pu.Terran.SCV])

  def select_all_barracks(self):
    return self.select_random_unit(
        [pu.Terran.Barracks, pu.Terran.BarracksFlying], "select_all_type")

  def select_barracks(self, location):
    return self.select_random_unit(
        [pu.Terran.Barracks, pu.Terran.BarracksFlying])

  def select_barracks_techlab(self):
    return self.select_random_unit([pu.Terran.BarracksTechLab])

  def select_engineering_bay(self):
    return self.select_random_unit([pu.Terran.EngineeringBay])

  def select_army(self):
    if self.action_available(actions.FUNCTIONS.select_army.id):
      return actions.FUNCTIONS.select_army("select")

  def train_marine(self):
    if self.action_available(actions.FUNCTIONS.Train_Marine_quick.id):
      return actions.FUNCTIONS.Train_Marine_quick("now")

  def research_combat_shield(self):
    if self.action_available(actions.FUNCTIONS.Research_CombatShield_quick.id):
      self.researched_combat_shield = True
      return actions.FUNCTIONS.Research_CombatShield_quick("now")

  def harvest_gather_refinery(self):
    refineries = self.get_units_by_types([pu.Terran.Refinery])
    if self.action_available(actions.FUNCTIONS.Harvest_Gather_screen.id) \
        and len(refineries) > 0:
      refinery = self.get_random_unit(refineries)
      return actions.FUNCTIONS.Harvest_Gather_screen(
          "now", self.clean_screen_location(refinery.x, refinery.y))

  def harvest_gather_minerals(self):
    minerals = self.get_units_by_types([pu.Neutral.MineralField,
                                        pu.Neutral.MineralField750])
    if self.action_available(actions.FUNCTIONS.Harvest_Gather_screen.id) \
        and len(minerals) > 0:
      mineral = self.get_random_unit(minerals)
      return actions.FUNCTIONS.Harvest_Gather_screen(
          "now", self.clean_screen_location(mineral.x, mineral.y))

  def attack_minimap(self, x, y,
                     allow_attack_with_workers=True,
                     random_distance=0):

    if (not self.action_available(actions.FUNCTIONS.Harvest_Gather_screen.id)
        and self.action_available(actions.FUNCTIONS.Attack_minimap.id)):

      width_multiplier = (
          self.minimap_trimmed_width / self.attack_minimap_width)
      height_multiplier = (
          self.minimap_trimmed_height / self.attack_minimap_height)

      x = (x * width_multiplier) + (width_multiplier / 2)
      y = (y * height_multiplier) + (width_multiplier / 2)

      if random_distance > 0:
        x_distance = random.randint(-1, 1) * random_distance
        y_distance = random.randint(-1, 1) * random_distance

        x += x_distance * (width_multiplier / 2)
        y += y_distance * (height_multiplier / 2)

      x = round(x)
      y = round(y)

      x, y = self.mirror_location(x, y)

      return actions.FUNCTIONS.Attack_minimap("now", self.clean_minimap_location(y, x))

  def build_supply_depot(self, locations, margin=3):
    supply_depots = self.get_units_by_types([pu.Terran.SupplyDepot])
    if (self.action_available(actions.FUNCTIONS.Build_SupplyDepot_screen.id)
        and len(supply_depots) < len(locations)):
      for location in locations:
        if not self.has_unit_at_location(
            supply_depots, location[0], location[1], margin):
          return actions.FUNCTIONS.Build_SupplyDepot_screen(
              "now", [location[0], location[1]])

  def build_barracks(self, locations, margin=5):
    barracks = self.get_units_by_types([pu.Terran.Barracks])
    if (self.action_available(actions.FUNCTIONS.Build_Barracks_screen.id)
        and len(barracks) < len(locations)):
      for location in locations:
        if not self.has_unit_at_location(
            barracks, location[0], location[1], margin):
          return actions.FUNCTIONS.Build_Barracks_screen(
              "now", [location[0], location[1]])

  def build_engineering_bay(self, locations, margin=5):
    engineering_bays = self.get_units_by_types([pu.Terran.EngineeringBay])
    if (self.action_available(actions.FUNCTIONS.Build_EngineeringBay_screen.id)
        and len(engineering_bays) < len(locations)):
      for location in locations:
        if not self.has_unit_at_location(
            engineering_bays, location[0], location[1], margin):
          return actions.FUNCTIONS.Build_EngineeringBay_screen(
              "now", [location[0], location[1]])

  def build_refinery(self):
    geysers = self.get_units_by_types([pu.Neutral.VespeneGeyser])
    if (self.action_available(actions.FUNCTIONS.Build_Refinery_screen.id)
        and len(geysers) > 0):
      geyser = self.get_random_unit(geysers)
      return actions.FUNCTIONS.Build_Refinery_screen(
          "now", self.clean_screen_location(geyser.x, geyser.y))

  def research_infantry_armor(self):
    if self.action_available(
        actions.FUNCTIONS.Research_TerranInfantryArmor_quick.id):
      self.infantry_armor_level += 1
      return actions.FUNCTIONS.Research_TerranInfantryArmor_quick("now")

  def research_infantry_weapons(self):
    if self.action_available(
        actions.FUNCTIONS.Research_TerranInfantryWeapons_quick.id):
      self.infantry_weapons_level += 1
      return actions.FUNCTIONS.Research_TerranInfantryWeapons_quick("now")

  def build_techlab(self, locations, margin=5):
    techlabs = self.get_units_by_types([pu.Terran.TechLab,
                                        pu.Terran.BarracksTechLab])
    if (self.action_available(actions.FUNCTIONS.Build_TechLab_screen.id)
        and len(techlabs) < len(locations)):
      for location in locations:
        if not self.has_unit_at_location(
            techlabs, location[0], location[1], margin):
          return actions.FUNCTIONS.Build_TechLab_screen(
              "now", [location[0], location[1]])

  def build_reactor(self, locations, margin=5):
    reactors = self.get_units_by_types([pu.Terran.Reactor,
                                        pu.Terran.BarracksReactor])
    if (self.action_available(actions.FUNCTIONS.Build_Reactor_screen.id)
        and len(reactors) < len(locations)):
      for location in locations:
        if not self.has_unit_at_location(
            reactors, location[0], location[1], margin):
          return actions.FUNCTIONS.Build_Reactor_screen(
              "now", [location[0], location[1]])

  def build_command_center(self, locations, margin=5):
    command_centers = self.get_units_by_types([pu.Terran.CommandCenter])
    if (self.action_available(actions.FUNCTIONS.Build_CommandCenter_screen.id)
        and len(command_centers) < len(locations)):
      location = self.get_build_location(command_centers, locations, margin)
      if location:
        return actions.FUNCTIONS.Build_CommandCenter_screen("now", location)

  def get_build_location(self, units, locations, margin):
    for location in locations:
      if not self.has_unit_at_location(units, location[0], location[1], margin):
        return location
    return None

  def no_op(self):
    return actions.FUNCTIONS.no_op()

  def sleep(self, count):
    for i in range(0, count):
      self.queue_action("no_op")


class state_manager:
  screen_width = 84
  screen_height = 84
  minimap_width = 64
  minimap_height = 64

  minimap_x_min = 0
  minimap_x_max = 64
  minimap_y_min = 0
  minimap_y_max = 64
  minimap_trimmed_width = 64
  minimap_trimmed_height = 64

  state_attributes = []

  def __init__(self, mirror_map=False):
    self.mirror_map = mirror_map
    self.reset()

  def add_attribute(self, attribute, params=[], divisor=1):
    self.state_attributes.append((attribute, params, divisor))

  def observe(self, obs):
    self.obs = obs
    if obs.first():
      player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                            features.PlayerRelative.SELF).nonzero()
      self.base_top_left = (player_y.any() and player_y.mean() <= 31)

      # Trim non-terrain off the minimap
      height_y, height_x = obs.observation.feature_minimap.height_map.nonzero()
      self.minimap_y_min = height_y.min()
      self.minimap_y_max = height_y.max()
      self.minimap_x_min = height_x.min()
      self.minimap_x_max = height_x.max()
      self.minimap_trimmed_width = self.minimap_x_max - self.minimap_x_min
      self.minimap_trimmed_height = self.minimap_y_max - self.minimap_y_min

  def reset(self):
    self.researched_combat_shield = False
    self.base_top_left = None
    self.infantry_armor_level = 0
    self.infantry_weapons_level = 0

  def get_state(self):
    sv = []  # State values
    for attribute, params, divisor in self.state_attributes:
      result = getattr(self, attribute)(*params) / divisor
      sv = numpy.append(sv, result)
    return sv

  def count_units_by_types(self, unit_types):
    units = [unit for unit in self.obs.observation.feature_units
             if unit.unit_type in unit_types]
    return len(units)

  def command_center_count(self):
    return self.count_units_by_types([pu.Terran.CommandCenter])

  def supply_depot_count(self):
    return self.count_units_by_types([pu.Terran.SupplyDepot])

  def barracks_count(self):
    return self.count_units_by_types([pu.Terran.Barracks,
                                      pu.Terran.BarracksFlying])

  def idle_worker_count(self):
    return self.obs.observation.player.idle_worker_count

  def food_army(self):
    return self.obs.observation.player.food_army

  def food_available(self):
    return (self.obs.observation.player.food_cap -
            self.obs.observation.player.food_used)

  def mineral_count(self):
    return self.obs.observation.player.minerals

  def game_loop(self):
    return self.obs.observation.game_loop[0]

  def minimap_enemy_hot(self, resolution):
    return self.minimap_units_hot(features.PlayerRelative.ENEMY, resolution)

  def minimap_self_hot(self, resolution):
    return self.minimap_units_hot(features.PlayerRelative.SELF, resolution)

  def minimap_units_hot(self, player_relative, resolution):
    hot_squares = numpy.zeros(resolution * resolution)
    unit_y, unit_x = (self.obs.observation.feature_minimap.player_relative ==
                        player_relative).nonzero()
    for i in range(0, len(unit_y)):
      y = int(math.ceil((unit_y[i] + 1 - self.minimap_y_min) / (self.minimap_trimmed_height / resolution)))
      x = int(math.ceil((unit_x[i] + 1 - self.minimap_x_min) / (self.minimap_trimmed_width / resolution)))
      hot_squares[((y - 1) * resolution) + (x - 1)] = 1
    if not self.base_top_left:
      hot_squares = hot_squares[::-1]
    return hot_squares

  def minimap_enemy_density(self, resolution, divisor=None):
    return self.minimap_units_density(
        features.PlayerRelative.ENEMY, resolution, divisor)

  def minimap_self_density(self, resolution, divisor=None):
    return self.minimap_units_density(
        features.PlayerRelative.SELF, resolution, divisor)

  def minimap_units_density(self, player_relative, resolution, divisor=None):
    hot_squares = numpy.zeros(resolution * resolution)
    unit_y, unit_x = (self.obs.observation.feature_minimap.player_relative ==
                        player_relative).nonzero()
    for i in range(0, len(unit_y)):
      y = int(math.ceil((unit_y[i] + 1 - self.minimap_y_min) / (self.minimap_trimmed_height / resolution)))
      x = int(math.ceil((unit_x[i] + 1 - self.minimap_x_min) / (self.minimap_trimmed_width / resolution)))
      hot_squares[((y - 1) * resolution) + (x - 1)] += 1
    for i in range(0, len(hot_squares)):
      if divisor is None:
        hot_squares[i] = hot_squares[i] / (
            self.minimap_trimmed_height * self.minimap_trimmed_width / resolution)
      else:
        hot_squares[i] = math.ceil(hot_squares[i] / (
            self.minimap_trimmed_height * self.minimap_trimmed_width / resolution * divisor))
    if not self.base_top_left:
      hot_squares = hot_squares[::-1]
    return hot_squares


class action_manager:
  complex_actions = {}

  def add_complex_action(self, name, actions):
    self.complex_actions[name] = actions

    print(self.complex_actions)

  def queue_complex_action(self, name, ursadon):
    for action, params in self.complex_actions[name]:
      ursadon.queue_action(action, params)
