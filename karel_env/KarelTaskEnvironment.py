import pdb
from copy import copy

NORTH = "north"
WEST = "west"
SOUTH = "south"
EAST = "east"

LEFT = "left"
RIGHT = "right"

PUT = "put"
GET = "get"
GO = "go"

def _getTeleportNorthNeighbor(loc, obstacles):
    x = loc[0]
    y = loc[1]
    yTel = max([p[1] for p in obstacles if p[0] == x])
    return (x, yTel - 1)


def _getTeleportEastNeighbor(loc, obstacles):
    x = loc[0]
    y = loc[1]
    xTel = max([p[0] for p in obstacles if p[1] == y])
    return (xTel - 1, y)


def _prettyPrintDict(x, ident=0):
    tabs = "".join(["\t" for _ in range(ident)])
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return ("\n" + tabs).join(_prettyPrintDict(el, ident) for el in x)
    if isinstance(x, dict):
        return tabs.join("({}\n{}{}\n{})".format(k, tabs + "\t", _prettyPrintDict(x[k], ident + 1), tabs) for k in x)
    else:
        pdb.set_trace()


# def generate_problem(problem_description_json, domain_description_json, domain_name, problem_name):
#     playground_width = problem_description_json["width"]
#     playground_height = problem_description_json["height"]
#     definition = {}
#
#     problem = {"problem": problem_name}
#     domain = {":domain": domain_name}
#
#     objects = {":objects": ["x_{}_{} - location".format(i, j)
#                             for i in range(playground_width + 2)
#                             # we want to add an extra fiedl for the wall: wall is at 0, and to the right of widht
#                             for j in range(playground_height + 2)
#                             ]
#                }
#
#     robot_position = ["(robot-at x_{}_{})".format(domain_description_json["robot-position-start"][0],
#                                                   domain_description_json["robot-position"][1])]
#     robot_orientation = ["(robot-facing-{})".format(domain_description_json["robot-orientation"])]
#     obstacles = []
#     # init.append(robot_position)
#     # init.append(robot_orientation)
#
#     # obstacles from the json
#     all_obstacles = [tuple(list_pair) for list_pair in domain_description_json["obstacles"]]
#     # wall obstacles
#     for x in range(0, playground_width + 1):
#
#         all_obstacles.append((x, 0))
#         all_obstacles.append((x, playground_height + 1))
#     for y in range(0, playground_height + 1):
#         all_obstacles.append((0, y))
#         all_obstacles.append((playground_width + 1, y))
#
#     for obs in all_obstacles:
#         print("adding obstacle ({},{})".format(obs[0], obs[1]))
#         obstacles.append("(obstacle x_{}_{})".format(obs[0], obs[1]))
#     # init.append(obstacles)
#
#     east_neighbors = []
#     north_neighbors = []
#
#     east_teleport_neighbors = []
#     north_teleport_neighbors = []
#
#     ## neighbor relations
#     for x in range(1, playground_width + 1):
#         for y in range(1, playground_height + 1):
#
#             if (x, y) in all_obstacles:
#                 continue
#             if x + 1 <= playground_width:
#                 east_neighbors.append(((x, y), (x + 1, y)))
#                 (xTelEast, yTelEast) = _getTeleportEastNeighbor((x, y), all_obstacles)
#                 east_teleport_neighbors.append(((x, y), (xTelEast, yTelEast)))
#
#             if y + 1 <= playground_height:
#                 north_neighbors.append(((x, y), (x, y + 1)))
#                 (xTelNorth, yTelNorth) = _getTeleportNorthNeighbor((x, y), all_obstacles)
#                 north_teleport_neighbors.append(((x, y), (xTelNorth, yTelNorth)))
#
#     neighbors = []
#     for n in east_neighbors:
#         neighbors.append("(east-neighbor x_{}_{} x_{}_{})".format(n[0][0], n[0][1], n[1][0], n[1][1]))
#     for n in north_neighbors:
#         neighbors.append("(north-neighbor x_{}_{} x_{}_{})".format(n[0][0], n[0][1], n[1][0], n[1][1]))
#     for n in east_teleport_neighbors:
#         neighbors.append("(teleport-east-neighbor x_{}_{} x_{}_{})".format(n[0][0], n[0][1], n[1][0], n[1][1]))
#     for n in north_teleport_neighbors:
#         neighbors.append("(teleport-north-neighbor x_{}_{} x_{}_{})".format(n[0][0], n[0][1], n[1][0], n[1][1]))
#
#     # starting cost
#     starting_cost = ["(= (total-cost) 0)"]
#     init = {":init": robot_position + robot_orientation + obstacles + neighbors + starting_cost}
#
#     # goal
#     marked_fields = ["(marked x_{}_{})".format(m[0], m[1]) for m in problem_description_json["markings"]]
#
#     robot_final_position = ["(robot-at x_{}_{})".format(problem_description_json["robot-position"][0],
#                                                         problem_description_json["robot-position"][1])]
#     robot_final_orientation = ["(robot-facing-{})".format(problem_description_json["robot-orientation"])]
#     goal = {":goal": {"and": marked_fields + robot_final_position + robot_final_orientation}}
#
#     metric = {":metric": "minimize (total-cost)"}
#
#     definition["define"] = [problem, domain, objects, init, goal, metric]
#
#     return _prettyPrintDict(definition)
#
#
# def generate_problem_model_2(problem_description_json, domain_description_json, domain_name, problem_name):
#     playground_width = problem_description_json["width"]
#     playground_height = problem_description_json["height"]
#     definition = {}
#
#     problem = {"problem": problem_name}
#     domain = {":domain": domain_name}
#
#     objects = {":objects": ["x{} - coordinate".format(i)
#                             for i in range(playground_width + 2)
#                             # we want to add an extra fiedl for the wall: wall is at 0, and to the right of widht
#                             ] + ["y{} - coordinate".format(j) for j in range(playground_height + 2)]
#                }
#
#     robot_position = ["(robot-at x{} y{})".format(domain_description_json["robot-position"][0],
#                                                   domain_description_json["robot-position"][1])]
#     robot_orientation = ["(robot-facing-{})".format(domain_description_json["robot-orientation"])]
#     obstacles = []
#     obstacle_relations = []
#
#     # obstacles from the json
#     all_obstacles = [tuple(list_pair) for list_pair in domain_description_json["obstacles"]]
#     # wall obstacles
#     for x in range(1, playground_width + 1):
#         all_obstacles.append((x, 0))
#         all_obstacles.append((x, playground_height + 1))
#     for y in range(1, playground_height + 1):
#         all_obstacles.append((0, y))
#         all_obstacles.append((playground_width + 1, y))
#
#     for obs in all_obstacles:
#         x = obs[0]
#         y = obs[1]
#         obstacles.append("(obstacle x{} y{})".format(x, y))
#         x_left = x - 1
#         x_right = x + 1
#         y_up = y + 1
#         y_down = y - 1
#         if x_left > 0:  # greater than zero because 0 is anyway an obstacle
#             obstacle_relations.append("(obstacle-to-east x{} y{})".format(x_left, y))
#         if x_right <= playground_width:
#             obstacle_relations.append("(obstacle-to-west x{} y{})".format(x_right, y))
#         if y_down > 0:
#             obstacle_relations.append("(obstacle-to-north x{} y{})".format(x, y_down))
#         if y_up <= playground_height:
#             obstacle_relations.append("(obstacle-to-south x{} y{})".format(x, y_up))
#
#     # succ and lt relations
#     succ_relations = []
#     lt_relations = []
#     for x in range(0, playground_width + 2):
#         if x - 1 > 0:
#             succ_relations.append("(succ x{} x{})".format(x - 1, x))
#         for lt_x in range(x):
#             lt_relations.append("(lt x{} x{})".format(lt_x, x))
#
#     for y in range(0, playground_height + 2):
#         if y - 1 > 0:
#             succ_relations.append("(succ y{} y{})".format(y - 1, y))
#         for lt_y in range(y):
#             lt_relations.append("(lt y{} y{})".format(lt_y, y))
#
#     # starting cost
#     starting_cost = ["(= (total-cost) 0)"]
#     init = {
#         ":init": robot_position + robot_orientation + obstacles + obstacle_relations + succ_relations + lt_relations + starting_cost}
#
#     # goal
#     marked_fields = ["(marked x{} y{})".format(m[0], m[1]) for m in problem_description_json["markings"]]
#
#     robot_final_position = ["(robot-at x{} y{})".format(problem_description_json["robot-position"][0],
#                                                         problem_description_json["robot-position"][1])]
#     robot_final_orientation = ["(robot-facing-{})".format(problem_description_json["robot-orientation"])]
#     goal = {":goal": {"and": marked_fields + robot_final_position + robot_final_orientation}}
#
#     metric = {":metric": "minimize (total-cost)"}
#
#     definition["define"] = [problem, domain, objects, init, goal, metric]
#
#     return _prettyPrintDict(definition)




def _grid_repr(height, width, robot_position, obstacles, gold_fields, robot_orientation, origin = "top_left"):

    orientation_symbol = {"west": '<', "north": '^', "south": 'v', "east": '>'}
    s = ""

    if origin == "bottom_left":
        row_range = range(height, 0, -1)
    elif origin == "top_left":
        row_range = range(1, height+1)

    for row in row_range:
        # the 2*width+1 is because of the visualization borders
        for _ in range(1, 2 * width + 1):
            s = s + "_"
        s = s + "\n"
        for col in range(1, width + 1):
            s = s + "|"
            #coord = (row, col)
            coord = (col, row)
            if coord == robot_position:
                if robot_orientation is not None:
                    s = s + orientation_symbol[robot_orientation]
                else:
                    s = s + "*"
            elif coord in obstacles:
                s = s + "x"
            elif coord in gold_fields:
                s = s + "g"
            else:
                s = s + " "
        s = s + "|"
        s = s + "\n"
    for _ in range(1, 2 * width + 1):
        s = s + "_"
    return s

def _new_orientation(old_orientation, direction):
    num2orientation = {0: NORTH, 1: WEST, 2: SOUTH, 3: EAST}
    orientation2num = {NORTH: 0, WEST: 1, SOUTH: 2, EAST: 3}
    if direction == LEFT:
        dir_num = 1
    elif direction == RIGHT:
        dir_num = -1
    else:
        raise ValueError("non existing direction")

    new_orientation = num2orientation[ (orientation2num[old_orientation] + dir_num) % 4 ]

    return new_orientation

def _get_next_step_position(position, orientation):
    x, y = position
    if orientation == NORTH:
        return (x, y+1)
    elif orientation == SOUTH:
        return (x, y-1)
    elif orientation == WEST:
        return (x-1, y)
    elif orientation == EAST:
        return (x + 1, y)
    else:
        raise ValueError("non-existing orientation {}".format(orientation))


class KarelTaskEnvironment:
    def __init__(self, problem_description_json):
        self.playground_width = problem_description_json["width"]
        self.playground_height = problem_description_json["height"]

        self.robot_position_start = tuple(problem_description_json["robot-position-start"])
        self.robot_orientation_start = problem_description_json["robot-orientation-start"]

        self.all_obstacles = {tuple(list_pair) for list_pair in problem_description_json["obstacles"]}
        # wall obstacles: it has additional +1 bcs we want to put "walls" as obstacles
        for x in range(0, self.playground_width + 1 + 1):
            self.all_obstacles.add((x, 0))
            self.all_obstacles.add((x, self.playground_height + 1))
        for y in range(0, self.playground_height + 1 + 1):
            self.all_obstacles.add((0, y))
            self.all_obstacles.add((self.playground_width + 1, y))

        self.gold_fields_start = {(field[0], field[1]) for field in problem_description_json["gold-start"]}
        self.poisoned_fields = [(m[0], m[1]) for m in
                                problem_description_json["poison"]] if "poison" in problem_description_json else []

        self.robot_carries_gold_start = True \
            if "robot-carries-gold-start" in problem_description_json and problem_description_json[
            "robot-carries-gold-start"] == "true" else False

        self.robot_carries_gold_end = True \
            if "robot-carries-gold-end" in problem_description_json and problem_description_json[
            "robot-carries-gold-end"] == "true" else False

        self.gold_fields_end = {(field[0], field[1]) for field in problem_description_json[
            "gold-end"]} if "gold-end" in problem_description_json else set()


        self.robot_position_end = tuple(problem_description_json["robot-position-end"]) \
            if "robot-position-end" in problem_description_json \
            else None

        self.robot_orientation_end = problem_description_json["robot-orientation-end"] \
            if "robot-orientation-end" in problem_description_json \
            else None

        self.home = tuple(problem_description_json["home"]) if "home" in problem_description_json else None
        self.hints = problem_description_json["hints"] if "hints" in problem_description_json else None
        self.hintsDomainConstants = problem_description_json["hints-constants"] \
            if "hints-constants" in problem_description_json else []


    def export_problem_hoc(self, domain_name, problem_name, use_hints=True):
        definition = {}

        problem = {"problem": problem_name}
        domain = {":domain": domain_name}

        objects = {":objects": ["x{} - coordinate".format(i)
                                for i in range(self.playground_width + 2)
                                # we want to add an extra field for the wall: wall is at 0, and to the right of width
                                ] + ["y{} - coordinate".format(j) for j in range(self.playground_height + 2)]
                               + ["{} - direction".format(d) for d in ["north", "south", "west", "east"]]
                   }

        robot_position_start = ["(robot-at x{} y{})".format(*self.robot_position_start)]

        robot_orientation_start = ["(robot-facing {})".format(self.robot_orientation_start)]


        # this is a constant initial state (robot always starts as healthy)
        robot_healthy = ["(robot-healthy)"]

        obstacles = []
        obstacle_relations = []

        obstacle_facing_init = []
        if _get_next_step_position(self.robot_position_start, self.robot_orientation_start) in self.all_obstacles:
            obstacle_facing_init = ["(robot-facing-obstacle)"]



        if self.home is not None:
            x_home, y_home = self.home
            home_location = ["(home x{} y{})".format(x_home, y_home)]
        else:
            home_location = []

        for obs in self.all_obstacles:
            x, y = obs
            obstacles.append("(obstacle x{} y{})".format(x, y))
            x_left = x - 1
            x_right = x + 1
            y_up = y + 1
            y_down = y - 1
            if x_left > 0:  # greater than zero because 0 is anyway an obstacle
                obstacle_relations.append("(obstacle-neighbor x{} y{} east)".format(x_left, y))
            if x_right <= self.playground_width:
                obstacle_relations.append("(obstacle-neighbor x{} y{} west)".format(x_right, y))
            if y_down > 0:
                obstacle_relations.append("(obstacle-neighbor x{} y{} north)".format(x, y_down))
            if y_up <= self.playground_height:
                obstacle_relations.append("(obstacle-neighbor x{} y{} south)".format(x, y_up))

        # succ and lt relations
        succ_relations = []

        for x in range(0, self.playground_width + 2):
            if x - 1 > 0:
                succ_relations.append("(succ x{} x{})".format(x - 1, x))

        for y in range(0, self.playground_height + 2):
            if y - 1 > 0:
                succ_relations.append("(succ y{} y{})".format(y - 1, y))

        gold_fields_start = ["(gold x{} y{})".format(m[0], m[1]) for m in self.gold_fields_start]
        poisoned_fields = ["(poison x{} y{}".format(m[0], m[1]) for m in self.poisoned_fields]

        robot_carries_gold_start = ["(robot-carries-gold)"] if self.robot_carries_gold_start is True else []

        rotation_rules = ["(rotation-left-neighbor north west)",
                          "(rotation-left-neighbor east north)",
                          "(rotation-left-neighbor south east)",
                          "(rotation-left-neighbor west south) ",
                          "(rotation-right-neighbor south west)",
                          "(rotation-right-neighbor west north)",
                          "(rotation-right-neighbor north east)",
                          "(rotation-right-neighbor east south)"
                          ]

        if self.hints is None or use_hints is False:
            hints_start = []
        else:
            hints_start = ["(hint{}-do)".format(hintId) for hintId, _ in enumerate(self.hints)]
            
        if self.hints is None or use_hints is False:
            hints_end = []
        else:
            hints_end = ["(hint{}-done)".format(hintId) for hintId, _ in enumerate(self.hints)]

        init = {":init": rotation_rules + robot_position_start + robot_orientation_start + robot_healthy +
                     obstacles + gold_fields_start + obstacle_relations + succ_relations +
                     poisoned_fields + robot_carries_gold_start + hints_start + home_location + obstacle_facing_init}

        gold_fields_end = ["(gold x{} y{})".format(m[0], m[1]) for m in self.gold_fields_end]

        removed_gold_fields = ["(not-gold x{} y{})".format(m[0], m[1]) for m in
                               self.gold_fields_start.difference(self.gold_fields_end)]
        robot_final_position = ["(robot-at x{} y{})".format(*self.robot_position_end)] \
            if not self.robot_position_end is None \
            else []

        robot_final_orientation = ["(robot-facing {})".format(self.robot_orientation_end)] \
            if not self.robot_orientation_end is None \
            else []

        robot_carries_gold_end = ["(robot-carries-gold)"] if self.robot_carries_gold_end is True else []

        goal = {":goal": {
            "and": gold_fields_end + removed_gold_fields + robot_final_position + robot_final_orientation +
                   robot_healthy + robot_carries_gold_end + hints_end}}

        definition["define"] = [problem, domain, objects, init, goal]

        return _prettyPrintDict(definition)


    @classmethod
    def export_domain_hoc(cls, domain_name="karel-hoc-domain", hints=None, generalizedPlanner=False, hintDomainConstants = {},
                          origin="top_left"):
        if origin == "top_left":
            progressDirection = "south"
            regressDirection = "north"
        else:
            progressDirection = "north"
            regressDirection = "south"
        hintDomainConstantsString = ""
        if hints is None:
            hintAction = ""
            hintPredicates = ""

        else:
            hintPredicatesToDo = "\n\t\t".join(["(hint{}-do)".format(hintId) for hintId, _ in enumerate(hints)])
            hintPredicatesDone = "\n\t\t".join(["(hint{}-done)".format(hintId) for hintId, _ in enumerate(hints)])
            hintPredicates = hintPredicatesToDo + "\n\t\t" + hintPredicatesDone
            if generalizedPlanner is True:
                hintDomainConstantsList = ["{} - {}".format(k, hintDomainConstants[k]) for k in hintDomainConstants]
                hintDomainConstantsString = "\n".join(hintDomainConstantsList)
            hintExpressions = ["""
            (:action hintex{hintId}
                :parameters
                ()
                :precondition
                (and
                    (hint{hintId}-do)
                    {hintCondition}
                )
                :effect
                (and
                  (hint{hintId}-done)
                )
              )
              """.format(hintCondition=hintEx, hintId = hintId) for (hintId, hintEx) in enumerate(hints)]
            hintAction ="\n".join(hintExpressions)

        # there is a syntax difference between generalized planner and the FD planner: a generalized planner needs
        # the declaration of constants, while the FD does not allow it
        if generalizedPlanner is True:
            constantsDeclaration = """      
            (:constants
                north
                south
                west
                east - direction
                {}
            )
            """.format(hintDomainConstantsString)
        else:
            constantsDeclaration = ""

        return """
    (define (domain {domainName})
      (:requirements :typing :action-costs :equality :adl :conditional-effects)
      (:types
            coordinate - integer
            direction - object 
      )

      (:predicates 
         (robot-at ?x ?y - coordinate)
         (robot-facing ?d - direction)
         (succ ?x ?y - integer)     
         (obstacle ?x ?y - coordinate)
         (obstacle-neighbor ?x ?y - coordinate ?d - direction)
         (obstacle-in-front)
         (rotation-left-neighbor ?d1 ?d2 - direction)
         (rotation-right-neighbor ?d1 ?d2 - direction)
         (gold ?x ?y - coordinate) 
         (not-gold ?x ?y - coordinate)    
         (poison ?x ?y - coordinate)
         (home ?x ?y - coordinate)
         (robot-carries-gold)
         (robot-poisoned)    
         (robot-healthy) 
         (robot-at-gold) 
         (robot-home)
         (robot-facing-obstacle) 
         {hintPredicate}   
      )
      
      {constantsDeclaration}
      
      {hintAction}
      (:action left
        :parameters
        (?directionStart ?directionEnd - direction ?x ?y - coordinate)
        :precondition
        (and
          (robot-facing ?directionStart)
          (rotation-left-neighbor ?directionStart ?directionEnd)
          (robot-at ?x ?y)
        )
        :effect
        (and
          (not (robot-facing ?directionStart))
          (robot-facing ?directionEnd)
          (when
            (obstacle-neighbor ?x ?y ?directionEnd)
            (obstacle-in-front)
          )
          (when
            (not (obstacle-neighbor ?x ?y ?directionEnd))
            (not (obstacle-in-front))
          )

        )
      )


      (:action right
        :parameters
        (?directionStart ?directionEnd - direction ?x ?y - coordinate)
        :precondition
        (and
          (robot-facing ?directionStart)
          (rotation-right-neighbor ?directionStart ?directionEnd)
          (robot-at ?x ?y)
        )
        :effect
        (and
          (not (robot-facing ?directionStart))
          (robot-facing ?directionEnd)

          (when
            (obstacle-neighbor ?x ?y ?directionEnd)
            (obstacle-in-front)
          )
          (when
            (not (obstacle-neighbor ?x ?y ?directionEnd))
            (not (obstacle-in-front))
          )

        )
      )


      (:action go
        :parameters
        (?xA ?yA ?xB ?yB - coordinate)
        :precondition
        (and
          (robot-at ?xA ?yA)      
          (not (obstacle ?xB ?yB))

          (or
            (and
              (robot-facing {goProgressDirection})                  
              (succ ?yA ?yB)
              (= ?xA ?xB)
            )
            (or
              (and
                (robot-facing {goRegressDirection})
                (succ ?yB ?yA)
                (= ?xA ?xB)
              )
              (or
                (and
                  (robot-facing west)
                  (succ ?xB ?xA)
                  (= ?yA ?yB)
                )
                (and
                  (robot-facing east)
                  (succ ?xA ?xB)
                  (= ?yA ?yB)
                )
              )
            )
          )


        )
        :effect
        (and
          (robot-at ?xB ?yB)
          (not (robot-at ?xA ?yA))
          (when
            (gold ?xB ?yB)
            (robot-at-gold)
           )

           (when
             (home ?xB ?yB)
             (robot-home)
           )
           (when
                (not (home ?xB ?yB))
                (not (robot-home))
           )

          (when
            (poison ?xB ?yB)
            (and
              (robot-poisoned)
              (not (robot-healthy))
            )
          )
        )    
      )


      (:action get
        :parameters
        (?x ?y - coordinate)
        :precondition
        (and
          (robot-at ?x ?y)
          (gold ?x ?y)
        )
        :effect
        (and
          (robot-carries-gold)
          (not (gold ?x ?y))
          (not-gold ?x ?y) 
        )
      )


      (:action put
        :parameters
        (?x ?y - coordinate)
        :precondition
        (and
          (robot-at ?x ?y)          
        )
        :effect
        (and
          (not (robot-carries-gold))
          (gold ?x ?y)          
        )
      )


    )
        """.format(domainName=domain_name, hintAction=hintAction,
                   hintPredicate=hintPredicates, constantsDeclaration=constantsDeclaration,
                   goProgressDirection=progressDirection, goRegressDirection=regressDirection)

    def _get_state_predicates(self, gold, position, orientation):
        preds = set()

        # check if the robot is on the gold field
        if position in gold:
            preds.add("gold")

        # check if the robot is at home
        if self.home is not None and position == self.home:
            preds.add("home")

        # add the orientation
        preds.add(orientation)

        x,y = position
        next_step_position = _get_next_step_position(position, orientation)
        # check if there is an obstacle in front of the robot
        if next_step_position in self.all_obstacles:
            preds.add("obstacle")

        return preds


    def extract_predicates_from_actions(self, action_seq):
        predicates = []


        gold_fields = copy(self.gold_fields_start)
        curr_position = self.robot_position_start
        curr_orientation = self.robot_orientation_start
        init_state_predicates = self._get_state_predicates(gold_fields, curr_position, curr_orientation)
        predicates.append(init_state_predicates)

        for action in action_seq:
            if action == PUT:
                gold_fields.add(curr_position)
            elif action == GET:
                # because this is the plan from the planner, it is guaranteed that there will be gold here
                try:
                    gold_fields.remove(curr_position)
                except:
                    raise ValueError("trying to get gold from {}, where there is no gold".format(curr_position))
            elif action == GO:
                curr_position = _get_next_step_position(curr_position, curr_orientation)
            elif action == LEFT or action == RIGHT:
                curr_orientation = _new_orientation(curr_orientation, action)
            else:
                raise ValueError("Invalid action {}".format(action))

            state_predicates = self._get_state_predicates(gold_fields, curr_position, curr_orientation)
            predicates.append(state_predicates)

        return predicates



    def __str__(self):

        s = ""
        s = s + "start:\n"
        s = s + _grid_repr(self.playground_height, self.playground_width, self.robot_position_start, self.all_obstacles,
                           self.gold_fields_start, self.robot_orientation_start)
        s = s + "\n\n\nend:\n"
        s = s + _grid_repr(self.playground_height, self.playground_width, self.robot_position_end, self.all_obstacles,
                           self.gold_fields_end, self.robot_orientation_end)
        s = s + "\n"

        return s
