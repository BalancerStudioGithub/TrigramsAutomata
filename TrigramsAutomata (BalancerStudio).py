import os
import random
import sys
import time
import threading
from enum import Enum
from collections import Counter
import unicodedata
import math 

# --- 导入 colorama 用于彩色输出 ---
try:
    from colorama import Fore, Back, Style, init
    init() 
    COLOR_SUPPORT = True 
except ImportError:
    COLOR_SUPPORT = False 
    class NoColor:
        def __getattr__(self, name):
            return ''
    Fore = NoColor()
    Back = NoColor()
    Style = NoColor()
    print("未找到 Colorama。输出将不带颜色。")


# --- 导入 pynput 用于监听键盘和鼠标事件 ---
PYNPUT_SUPPORT = False 
try:
    from pynput import keyboard, mouse
    PYNPUT_SUPPORT = True 
except ImportError:
    print(Fore.RED + "警告：未找到 'pynput' 库。键盘快捷键如 'q', 'p', '+/-' 将无法使用，鼠标点击功能也将不可用。游戏将需要按 'Enter' 键才能进行下一步。请运行 'pip install pynput' 以启用完整功能。" + Style.RESET_ALL)
except Exception as e:
    print(Fore.RED + f"警告：导入 'pynput' 时发生错误：{e}。完整功能已禁用。" + Style.RESET_ALL)


# --- 辅助函数：获取显示宽度 ---
def get_display_width(text: str) -> int:
    """
    计算字符串的显示宽度，考虑宽字符和ANSI颜色代码。
    例如，一个中文字符通常占用两个英文字符的宽度。
    """
    width = 0
    in_color_code = False 
    i = 0
    while i < len(text):
        if text[i] == '\x1b' and i + 1 < len(text) and text[i+1] == '[':
            in_color_code = True
            i += 2
        elif in_color_code and text[i] == 'm':
            in_color_code = False
            i += 1
            continue

        if in_color_code: 
            i += 1
            continue

        if unicodedata.east_asian_width(text[i]) in ('F', 'W', 'A'): 
            width += 2 
        else:
            width += 1 
        i += 1
    return width

# --- 1. 卦象枚举定义 ---
class Trigram(Enum):
    """表示八卦符号和空状态。"""
    QIAN = (1, 1, 1) # 乾 (天，创造力) - 阳 
    KUN = (0, 0, 0)  # 坤 (地，接受性) - 阴
    ZHEN = (0, 1, 0) # 震 (雷，震动)
    XUN = (0, 0, 1)  # 巽 (风/木，柔顺)
    KAN = (1, 0, 0)  # 坎 (水，险陷)
    LI = (1, 1, 0)   # 离 (火，依附)
    GEN = (1, 0, 1)  # 艮 (山，静止)
    DUI = (0, 1, 1)  # 兑 (泽/沼，喜悦)
    EMPTY = (-1, -1, -1) # 表示被湮灭的单元格（空）

# --- 2. 卦象符号 (Unicode 和 备用) ---
# 用于视觉表示的Unicode符号
GUA_UNICODE = {
    Trigram.QIAN: '☰',
    Trigram.KUN: '☷',
    Trigram.ZHEN: '☳',
    Trigram.XUN: '☴',
    Trigram.KAN: '☵',
    Trigram.LI: '☲',
    Trigram.GEN: '☶',
    Trigram.DUI: '☱',
    Trigram.EMPTY: '　' # 全角空格表示空单元格
}

# 备用中文字符，用于在Unicode显示不佳的环境中
FALLBACK_CHINESE = {
    Trigram.QIAN: '乾',
    Trigram.KUN: '坤',
    Trigram.ZHEN: '震',
    Trigram.XUN: '巽',
    Trigram.KAN: '坎',
    Trigram.LI: '离',
    Trigram.GEN: '艮',
    Trigram.DUI: '兑',
    Trigram.EMPTY: '空'
}

# 每个卦象的颜色，用于终端彩色输出
TRIGRAM_COLORS = {
    Trigram.QIAN: Fore.LIGHTYELLOW_EX,
    Trigram.KUN: Fore.YELLOW,
    Trigram.ZHEN: Fore.GREEN,
    Trigram.XUN: Fore.CYAN,
    Trigram.KAN: Fore.BLUE,
    Trigram.LI: Fore.RED,
    Trigram.GEN: Fore.LIGHTBLACK_EX,
    Trigram.DUI: Fore.MAGENTA,
    Trigram.EMPTY: Style.DIM + Fore.BLACK
}

# --- 3. 元素枚举定义 ---
class Element(Enum):
    """表示五行元素。"""
    WOOD = 1 
    FIRE = 2 
    EARTH = 3 
    METAL = 4 
    WATER = 5 

# --- 4. 卦象到元素的映射 (常见关联) ---
# 定义每个卦象的元素关联
TRIGRAM_TO_ELEMENT = {
    Trigram.QIAN: Element.METAL,
    Trigram.KUN: Element.EARTH,
    Trigram.ZHEN: Element.WOOD,
    Trigram.XUN: Element.WOOD,
    Trigram.KAN: Element.WATER,
    Trigram.LI: Element.FIRE,
    Trigram.GEN: Element.EARTH,
    Trigram.DUI: Element.METAL,
}

# 创建从元素到相关联卦象列表的反向映射
ELEMENT_TO_TRIGRAMS = {element: [] for element in Element}
for trigram, element in TRIGRAM_TO_ELEMENT.items():
    ELEMENT_TO_TRIGRAMS[element].append(trigram)

# --- 5. 元素相生关系 (生产) ---
# 定义谁生谁 (例如，木生火)
ELEMENT_PRODUCE = {
    Element.WOOD: Element.FIRE,
    Element.FIRE: Element.EARTH,
    Element.EARTH: Element.METAL,
    Element.METAL: Element.WATER,
    Element.WATER: Element.WOOD,
}

# --- 5.1 元素相生关系 (反向映射，用于“生我者”保护) ---
# 定义谁被谁生 (例如，火被木生)
ELEMENT_PRODUCED_BY = {v: k for k, v in ELEMENT_PRODUCE.items()}


# --- 6. 元素相克关系 (消耗/克制) ---
# 定义谁克谁 (例如，木克土)
ELEMENT_CONSUME = {
    Element.WOOD: Element.EARTH,
    Element.EARTH: Element.WATER,
    Element.WATER: Element.FIRE,
    Element.FIRE: Element.METAL,
    Element.METAL: Element.WOOD,
}

# --- 7. 辅助函数：将像素坐标转换为网格坐标 (近似) ---
# 这些是终端字符的近似像素值，用于鼠标交互
CHAR_WIDTH_PIXELS = 20 
CHAR_HEIGHT_PIXELS = 40 
GRID_TOP_OFFSET_PIXELS = 80 
GRID_LEFT_OFFSET_PIXELS = 0 

def get_grid_coords_from_pixels(x_pixel: int, y_pixel: int, char_width_pixels: int, char_height_pixels: int, grid_left_offset: int, grid_top_offset: int, cell_char_width: int = 2) -> tuple[int, int]:
    """
    将鼠标点击的像素坐标转换为网格 (行, 列) 坐标。
    """
    adjusted_x = x_pixel - grid_left_offset 
    adjusted_y = y_pixel - grid_top_offset 
    col = int(adjusted_x / (char_width_pixels * cell_char_width)) 
    row = int(adjusted_y / char_height_pixels) 
    return row, col

# --- 8. 游戏配置 ---
class GameConfig:
    """游戏参数的集中配置。"""
    WIDTH = 60 
    HEIGHT = 40 
    UPDATE_INTERVAL = 0.5 
    INITIAL_DENSITY = 0.1 
    CELL_DISPLAY_WIDTH = 2 
    EMPTY_REBIRTH_CYCLES_MIN = 100000000 
    EMPTY_REBIRTH_CYCLES_MAX = 15000000000000 
    BI_SHENG_SAN_PROBABILITY = 0.1 
    BI_SHENG_SAN_DELAY_MIN = 0.01 
    BI_SHENG_SAN_DELAY_MAX = 0.02 
    ANNIHILATION_THRESHOLD = 1 
    REPETITION_ANNIHILATION_THRESHOLD = 3 
    MIN_TRIGRAM_LIFESPAN = 0.001 
    MAX_TRIGRAM_LIFESPAN = 0.02 

# --- 新增 CellState 枚举和 Cell 类 ---
class CellState(Enum):
    """定义单元格的瞬态。"""
    NORMAL = 1          
    FLASHING = 2        
    ANNIHILATING = 3    

class Cell:
    """表示八卦网格中的一个单元格。"""
    def __init__(self, trigram: Trigram):
        self.trigram = trigram 
        self.state = CellState.NORMAL 
        self.flash_counter = 0      
        self.empty_counter = 0      
        self.rebirth_target_cycle = 0 
        self.birth_time = None      
        self.creation_time = time.time() 

# --- 9. 八卦网格类 ---
class BaguaGrid:
    """管理八卦网格及其演化。"""
    def __init__(self, width: int, height: int):
        self.width = width 
        self.height = height 
        self.grid = [[Cell(Trigram.KUN) for _ in range(width)] for _ in range(height)]
        self.pending_births = []
        self.initialize_grid()

    def initialize_grid(self):
        """初始化网格以形成太极（阴阳）符号。"""
        center_x = self.width / 2 
        center_y = self.height / 2 
        radius = min(self.width, self.height) / 2 - 1 

        small_circle_radius = radius / 2
        eye_radius = radius / 8

        for r in range(self.height):
            for c in range(self.width):
                x = c - center_x
                y = r - center_y

                distance = math.sqrt(x**2 + y**2) 

                if distance <= radius: 
                    is_yin_side = False 

                    dist_to_top_half_center = math.sqrt(x**2 + (y + small_circle_radius)**2)

                    dist_to_bottom_half_center = math.sqrt(x**2 + (y - small_circle_radius)**2)

                    if x >= 0: 
                        if dist_to_top_half_center <= small_circle_radius:
                            is_yin_side = False
                        elif dist_to_bottom_half_center <= small_circle_radius:
                            is_yin_side = True
                        else:
                            is_yin_side = False
                    else: 
                        if dist_to_top_half_center <= small_circle_radius:
                            is_yin_side = False
                        elif dist_to_bottom_half_center <= small_circle_radius:
                            is_yin_side = True
                        else:
                            is_yin_side = True

                    # 处理“眼睛”
                    eye1_center_x = 0
                    eye1_center_y = -small_circle_radius
                    dist_to_eye1 = math.sqrt((x - eye1_center_x)**2 + (y - eye1_center_y)**2)

                    eye2_center_x = 0
                    eye2_center_y = small_circle_radius
                    dist_to_eye2 = math.sqrt((x - eye2_center_x)**2 + (y - eye2_center_y)**2)

                    if dist_to_eye1 <= eye_radius:
                        self.grid[r][c].trigram = Trigram.KUN 
                    elif dist_to_eye2 <= eye_radius:
                        self.grid[r][c].trigram = Trigram.QIAN 
                    elif is_yin_side:
                        self.grid[r][c].trigram = Trigram.KUN 
                    else:
                        self.grid[r][c].trigram = Trigram.QIAN 
                else:
                    self.grid[r][c].trigram = Trigram.EMPTY
                    self.grid[r][c].rebirth_target_cycle = random.randint(
                        GameConfig.EMPTY_REBIRTH_CYCLES_MIN, GameConfig.EMPTY_REBIRTH_CYCLES_MAX
                    )
                self.grid[r][c].creation_time = time.time()


    def get_neighbors_coords(self, r: int, c: int) -> list[tuple[int, int]]:
        """返回所有8个邻居 (包括对角线) 的网格坐标，并进行环绕处理。"""
        neighbors_coords = []
        for dr in [-1, 0, 1]: 
            for dc in [-1, 0, 1]: 
                if dr == 0 and dc == 0: 
                    continue
                nr, nc = (r + dr) % self.height, (c + dc) % self.width 
                neighbors_coords.append((nr, nc))
        return neighbors_coords

    def get_neighbor_trigrams(self, r: int, c: int, source_grid: list[list[Cell]]) -> list[Trigram]:
        """从给定网格中返回邻居卦象的列表。"""
        return [source_grid[nr][nc].trigram for nr, nc in self.get_neighbors_coords(r, c)]

    def calculate_next_cell_state(self, current_cell: Cell, neighbors_trigrams: list[Trigram]) -> tuple[Trigram, CellState]:
        """
        根据当前单元格的状态和邻居计算下一个卦象和状态。
        应用五行理论 (相生, 相克) 和单元格生命周期。
        """
        current_trigram = current_cell.trigram 

        # --- 处理 EMPTY 状态和重生 ---
        if current_trigram == Trigram.EMPTY:
            if current_cell.empty_counter >= current_cell.rebirth_target_cycle:
                all_non_empty_trigrams = [t for t in list(Trigram) if t != Trigram.EMPTY] 
                return random.choice(all_non_empty_trigrams), CellState.NORMAL 
            else:
                return Trigram.EMPTY, CellState.NORMAL 

        # --- 新增：卦象生命值检查 ---
        if current_cell.state == CellState.NORMAL:
            if current_cell.creation_time is not None and \
               (time.time() - current_cell.creation_time < GameConfig.MIN_TRIGRAM_LIFESPAN):
                return current_trigram, CellState.NORMAL

        # --- 处理 FLASHING/ANNIHILATING 状态 ---
        if current_cell.state == CellState.FLASHING:
            current_cell.flash_counter -= 0.1 
            if current_cell.flash_counter <= 0: 
                return current_trigram, CellState.ANNIHILATING 
            return current_trigram, CellState.FLASHING 

        if current_cell.state == CellState.ANNIHILATING:
            new_rebirth_target = random.randint(
                GameConfig.EMPTY_REBIRTH_CYCLES_MIN, GameConfig.EMPTY_REBIRTH_CYCLES_MAX
            )
            current_cell.rebirth_target_cycle = new_rebirth_target
            return Trigram.EMPTY, CellState.NORMAL 

        # --- 非空、非闪烁单元格的正常演化逻辑 ---
        current_element = TRIGRAM_TO_ELEMENT.get(current_trigram) 
        if not current_element: 
            return random.choice([t for t in list(Trigram) if t != Trigram.EMPTY]), CellState.NORMAL

        target_element_influence = {e: 0 for e in Element} 
        overcome_score_on_current = 0 

        for neighbor_trigram in neighbors_trigrams: 
            if neighbor_trigram == Trigram.EMPTY: 
                continue

            neighbor_element = TRIGRAM_TO_ELEMENT.get(neighbor_trigram) 
            if not neighbor_element: continue

            # 应用相生 (生产) 影响力来自邻居
            produced_element_by_neighbor = ELEMENT_PRODUCE.get(neighbor_element) 
            if produced_element_by_neighbor:
                target_element_influence[produced_element_by_neighbor] += 1 

            # 应用“生我者” (当前单元格的生产者) 保护
            producer_of_current = ELEMENT_PRODUCED_BY.get(current_element) 
            if producer_of_current and neighbor_element == producer_of_current:
                target_element_influence[current_element] += 1.0 

            # 应用相克 (消耗/克制) 影响力来自邻居
            for consumed_element, consumer_element in ELEMENT_CONSUME.items():
                if consumer_element == neighbor_element: 
                    target_element_influence[consumed_element] -= 1 
                    if consumed_element == current_element: 
                        overcome_score_on_current += 1 

        # 当前单元格的“惯性”
        target_element_influence[current_element] += 0.1 

        # 基于元素消耗进行湮灭检查
        if overcome_score_on_current >= GameConfig.ANNIHILATION_THRESHOLD:
            return current_trigram, CellState.FLASHING 

        # 确定主导元素以进行演化
        max_score = -float('inf') 
        dominant_elements = [] 
        for element, score in target_element_influence.items():
            if score > max_score:
                max_score = score
                dominant_elements = [element]
            elif score == max_score:
                dominant_elements.append(element)

        if current_element in dominant_elements:
            return current_trigram, CellState.NORMAL 
        else:
            if dominant_elements:
                chosen_dominant_element = random.choice(dominant_elements) 
            else:
                chosen_dominant_element = random.choice(list(Element)) 

            possible_trigrams = ELEMENT_TO_TRIGRAMS.get(chosen_dominant_element, []) 
            if possible_trigrams:
                return random.choice(possible_trigrams), CellState.NORMAL 
            else:
                return random.choice([t for t in list(Trigram) if t != Trigram.EMPTY]), CellState.NORMAL

    def _apply_repetition_annihilation(self, temp_grid: list[list[Cell]]):
        """
        检查并标记连续出现3个或更多相同卦象（水平或垂直方向）的单元格进行湮灭。
        直接修改 temp_grid。
        """
        annihilated_coords = set() 

        # 水平检查
        for r in range(self.height):
            for c in range(self.width):
                current_cell = temp_grid[r][c] 
                current_trigram = current_cell.trigram 

                if current_trigram == Trigram.EMPTY: 
                    continue

                if current_cell.state == CellState.NORMAL and \
                   current_cell.creation_time is not None and \
                   (time.time() - current_cell.creation_time < GameConfig.MIN_TRIGRAM_LIFESPAN):
                    continue

                if (r, c) in annihilated_coords and current_cell.state == CellState.FLASHING:
                    continue

                horizontal_streak = [] 
                for k in range(GameConfig.REPETITION_ANNIHILATION_THRESHOLD): 
                    if c + k < self.width: 
                        check_cell = temp_grid[r][c + k] 
                        if check_cell.trigram == current_trigram and \
                           (check_cell.state != CellState.NORMAL or \
                           (check_cell.creation_time is not None and \
                            time.time() - check_cell.creation_time >= GameConfig.MIN_TRIGRAM_LIFESPAN)):
                            horizontal_streak.append((r, c + k)) 
                        else:
                            break 
                    else:
                        break 

                if len(horizontal_streak) >= GameConfig.REPETITION_ANNIHILATION_THRESHOLD: 
                    for ar, ac in horizontal_streak: 
                        if temp_grid[ar][ac].state not in [CellState.FLASHING, CellState.ANNIHILATING]:
                            temp_grid[ar][ac].state = CellState.FLASHING 
                            temp_grid[ar][ac].flash_counter = 1 
                            annihilated_coords.add((ar, ac)) 

        # 垂直检查 (逻辑与水平检查类似)
        for c in range(self.width):
            for r in range(self.height):
                current_cell = temp_grid[r][c]
                current_trigram = current_cell.trigram

                if current_trigram == Trigram.EMPTY:
                    continue

                if current_cell.state == CellState.NORMAL and \
                   current_cell.creation_time is not None and \
                   (time.time() - current_cell.creation_time < GameConfig.MIN_TRIGRAM_LIFESPAN):
                    continue

                if (r, c) in annihilated_coords and current_cell.state == CellState.FLASHING:
                    continue

                vertical_streak = []
                for k in range(GameConfig.REPETITION_ANNIHILATION_THRESHOLD):
                    if r + k < self.height:
                        check_cell = temp_grid[r + k][c]
                        if check_cell.trigram == current_trigram and \
                           (check_cell.state != CellState.NORMAL or \
                           (check_cell.creation_time is not None and \
                            time.time() - check_cell.creation_time >= GameConfig.MIN_TRIGRAM_LIFESPAN)):
                            vertical_streak.append((r + k, c))
                        else:
                            break
                    else:
                        break 

                if len(vertical_streak) >= GameConfig.REPETITION_ANNIHILATION_THRESHOLD:
                    for ar, ac in vertical_streak:
                        if temp_grid[ar][ac].state not in [CellState.FLASHING, CellState.ANNIHILATING]:
                            temp_grid[ar][ac].state = CellState.FLASHING
                            temp_grid[ar][ac].flash_counter = 1
                            annihilated_coords.add((ar, ac))


    def update_grid(self):
        """
        更新整个网格一个周期。
        分两步进行计算：第一步是单个单元格的演化，
        第二步是“二生三”的生成，以确保状态一致。
        """
        # 第一步：根据当前网格计算每个单元格的下一个状态
        temp_new_grid_cells = [[None for _ in range(self.width)] for _ in range(self.height)]

        for r in range(self.height):
            for c in range(self.width):
                current_cell = self.grid[r][c] 
                neighbors_trigrams = self.get_neighbor_trigrams(r, c, self.grid)

                next_trigram, next_state = self.calculate_next_cell_state(current_cell, neighbors_trigrams) 

                new_cell = Cell(next_trigram) 
                new_cell.state = next_state 

                # --- 处理 new_cell 的 creation_time ---
                if next_trigram == current_cell.trigram and next_state == current_cell.state:
                    new_cell.creation_time = current_cell.creation_time
                elif next_trigram == Trigram.EMPTY:
                    new_cell.creation_time = None
                else:
                    new_cell.creation_time = time.time()


                if next_state == CellState.FLASHING:
                    new_cell.flash_counter = current_cell.flash_counter
                    if current_cell.state != CellState.FLASHING: 
                        new_cell.flash_counter = 1 
                elif current_cell.state == CellState.ANNIHILATING and next_state == CellState.NORMAL and next_trigram == Trigram.EMPTY:
                    new_cell.rebirth_target_cycle = current_cell.rebirth_target_cycle
                    new_cell.empty_counter = current_cell.empty_counter + 1
                elif next_trigram == Trigram.EMPTY:
                    new_cell.empty_counter = current_cell.empty_counter + 1
                    new_cell.rebirth_target_cycle = current_cell.rebirth_target_cycle
                else: 
                    new_cell.empty_counter = 0
                    new_cell.rebirth_target_cycle = 0

                new_cell.birth_time = current_cell.birth_time


                temp_new_grid_cells[r][c] = new_cell 

        # --- 对计算出的下一个网格状态应用重复湮灭规则 ---
        self._apply_repetition_annihilation(temp_new_grid_cells)

        # --- 第二步：对“临时计算”出的新网格应用“二生三”生成逻辑 ---
        current_cycle_birth_positions = set() 
        newly_scheduled_births = [] 

        for r in range(self.height):
            for c in range(self.width):
                if temp_new_grid_cells[r][c].trigram == Trigram.EMPTY:
                    continue

                if temp_new_grid_cells[r][c].state in [CellState.FLASHING, CellState.ANNIHILATING]:
                    continue

                current_cell_element = TRIGRAM_TO_ELEMENT.get(temp_new_grid_cells[r][c].trigram) 
                if not current_cell_element:
                    continue

                for nr, nc in self.get_neighbors_coords(r, c): 
                    neighbor_cell = temp_new_grid_cells[nr][nc] 
                    neighbor_element = TRIGRAM_TO_ELEMENT.get(neighbor_cell.trigram) 

                    if neighbor_element and ELEMENT_PRODUCE.get(current_cell_element) == neighbor_element:
                        if random.random() < GameConfig.BI_SHENG_SAN_PROBABILITY: 
                            produced_trigrams = ELEMENT_TO_TRIGRAMS.get(neighbor_element, []) 

                            if produced_trigrams: 
                                potential_birth_coords = list(set(
                                    self.get_neighbors_coords(r, c) + self.get_neighbors_coords(nr, nc)
                                ))

                                suitable_birth_coords = [] 
                                for br, bc in potential_birth_coords:
                                    target_temp_cell = temp_new_grid_cells[br][bc]
                                    if target_temp_cell.trigram == Trigram.EMPTY and \
                                       target_temp_cell.state not in [CellState.FLASHING, CellState.ANNIHILATING] and \
                                       (br, bc) not in current_cycle_birth_positions:
                                        suitable_birth_coords.append((br,bc))

                                if suitable_birth_coords: 
                                    birth_r, birth_c = random.choice(suitable_birth_coords) 
                                    new_trigram_for_birth = random.choice(produced_trigrams) 

                                    individual_birth_delay = random.uniform(GameConfig.BI_SHENG_SAN_DELAY_MIN, GameConfig.BI_SHENG_SAN_DELAY_MAX)
                                    scheduled_birth_time = time.time() + individual_birth_delay 

                                    newly_scheduled_births.append((new_trigram_for_birth, birth_r, birth_c, scheduled_birth_time)) 
                                    current_cycle_birth_positions.add((birth_r, birth_c)) 

        self.pending_births.extend(newly_scheduled_births) 

        # --- 应用预定的生成 ---
        current_time = time.time() 
        new_pending_births = [] 
        for trigram, r, c, scheduled_time in self.pending_births:
            if current_time >= scheduled_time: 
                target_cell = temp_new_grid_cells[r][c]
                if target_cell.trigram == Trigram.EMPTY and \
                   target_cell.state not in [CellState.FLASHING, CellState.ANNIHILATING]:
                    temp_new_grid_cells[r][c] = Cell(trigram) 
                    temp_new_grid_cells[r][c].state = CellState.NORMAL 
                    temp_new_grid_cells[r][c].empty_counter = 0 
                    temp_new_grid_cells[r][c].rebirth_target_cycle = 0 
                    temp_new_grid_cells[r][c].birth_time = current_time 
                    temp_new_grid_cells[r][c].creation_time = current_time 
            else:
                new_pending_births.append((trigram, r, c, scheduled_time)) 
        self.pending_births = new_pending_births 


        self.grid = temp_new_grid_cells

    def set_trigram_at(self, row: int, col: int, trigram: Trigram):
        """手动在指定网格位置设置一个卦象。重置其状态计数器。"""
        if 0 <= row < self.height and 0 <= col < self.width: 
            self.grid[row][col].trigram = trigram 
            self.grid[row][col].state = CellState.NORMAL 
            self.grid[row][col].flash_counter = 0 
            self.grid[row][col].empty_counter = 0 
            self.grid[row][col].rebirth_target_cycle = 0 
            self.grid[row][col].birth_time = None 
            self.grid[row][col].creation_time = time.time() 

    def get_trigram_at(self, row: int, col: int) -> Trigram:
        """检索指定网格位置的卦象。"""
        if 0 <= row < self.height and 0 <= col < self.width:
            return self.grid[row][col].trigram
        return None 

# --- 10. 游戏控制和显示 ---
class Game:
    """管理主游戏循环、显示和用户交互。"""
    def __init__(self):
        self.grid = BaguaGrid(GameConfig.WIDTH, GameConfig.HEIGHT) 
        self.running = True 
        self.paused = False 
        self.update_interval = GameConfig.UPDATE_INTERVAL 
        self.keyboard_listener = None 
        self.mouse_listener = None 
        self.trigrams_cycle = [t for t in list(Trigram) if t != Trigram.EMPTY]
        self.previous_grid_state = None  # 用于存储上一帧的卦象和状态，以便进行局部更新
        self.first_run = True # 标记是否是第一次运行，第一次需要清屏并绘制完整界面
        self.header_lines = 2 # 游戏顶部信息占用的行数

    def clear_screen(self):
        """清除终端屏幕。"""
        if os.name == 'nt': 
            _ = os.system('cls')
        else: 
            _ = os.system('clear')

    def print_grid(self):
        """使用颜色和符号将当前网格状态打印到控制台，并实现局部更新以减少闪烁。"""
        if self.first_run:
            self.clear_screen()
            self.previous_grid_state = [[(cell.trigram, cell.state) for cell in row] for row in self.grid.grid]
        
        # 移动光标到起始位置更新顶部信息
        sys.stdout.write(f"\033[H")  # Move cursor to home position
        sys.stdout.write(f"--- CellularAutomata of EightTrigrams (八卦相生) [BalancerStudio]---\n")
        sys.stdout.write(f"update: {self.update_interval:.2f}s | {'pause' if self.paused else 'run'} | ")
        sys.stdout.write(f"BI_SHENG_SAN_PROBABILITY: {GameConfig.BI_SHENG_SAN_PROBABILITY:.2f} | ")
        if PYNPUT_SUPPORT:
            sys.stdout.write("press 'q' quit, 'p' pause/go, '+/-' speed, '*/' prob, 'r' replay\n")
            sys.stdout.write("Click to change Trigram\n")
        else:
            sys.stdout.write("Press 'Enter' to proceed (pynput library not found)\n")
        
        # 移动光标到网格起始位置（跳过header_lines）
        grid_start_row = self.header_lines + 2 # 2 for the extra lines of instructions.
        
        # 进行局部更新
        for r in range(self.grid.height):
            for c in range(self.grid.width):
                cell = self.grid.grid[r][c]
                current_display_tuple = (cell.trigram, cell.state)

                # 如果单元格状态未改变，则不重绘，减少闪烁
                if not self.first_run and current_display_tuple == self.previous_grid_state[r][c]:
                    continue

                trigram = cell.trigram
                symbol_to_display = ""
                color_to_display = Style.RESET_ALL

                if cell.state == CellState.FLASHING:
                    symbol_to_display = FALLBACK_CHINESE.get(trigram, '??')
                    color_to_display = TRIGRAM_COLORS.get(trigram, Fore.WHITE) + Style.BRIGHT
                elif trigram == Trigram.EMPTY:
                    symbol_to_display = GUA_UNICODE.get(trigram, FALLBACK_CHINESE.get(trigram, '??'))
                    color_to_display = TRIGRAM_COLORS.get(trigram, Style.DIM + Fore.BLACK)
                else:
                    symbol_to_display = GUA_UNICODE.get(trigram, FALLBACK_CHINESE.get(trigram, '??'))
                    color_to_display = TRIGRAM_COLORS.get(trigram, Style.RESET_ALL)

                colored_symbol = f"{color_to_display}{symbol_to_display}{Style.RESET_ALL}"
                current_display_width = get_display_width(symbol_to_display)
                padding_needed = GameConfig.CELL_DISPLAY_WIDTH - current_display_width

                # 定位光标并打印
                sys.stdout.write(f"\033[{r + grid_start_row};{c * GameConfig.CELL_DISPLAY_WIDTH + 1}H")
                sys.stdout.write(colored_symbol + ' ' * padding_needed)

                # 更新previous_grid_state
                self.previous_grid_state[r][c] = current_display_tuple
        
        # 第一次运行后将first_run设为False
        if self.first_run:
            self.first_run = False
        
        sys.stdout.flush() # 强制刷新输出缓冲区

    def on_press(self, key):
        """pynput 的键盘事件处理函数。"""
        try:
            if key == keyboard.Key.esc or (hasattr(key, 'char') and key.char == 'q'): 
                self.running = False 
                return False 
            elif hasattr(key, 'char') and key.char == 'p': 
                self.paused = not self.paused 
            elif hasattr(key, 'char') and key.char == '+': 
                self.update_interval = max(0.001, self.update_interval - 0.1) 
            elif hasattr(key, 'char') and key.char == '-': 
                self.update_interval = min(1.0, self.update_interval + 0.1) 
            elif hasattr(key, 'char') and key.char == '*': 
                GameConfig.BI_SHENG_SAN_PROBABILITY = min(1.0, GameConfig.BI_SHENG_SAN_PROBABILITY + 0.02)
            elif hasattr(key, 'char') and key.char == '/': 
                GameConfig.BI_SHENG_SAN_PROBABILITY = max(0.001, GameConfig.BI_SHENG_SAN_PROBABILITY - 0.02)
            elif hasattr(key, 'char') and key.char == 'r': 
                self.grid = BaguaGrid(GameConfig.WIDTH, GameConfig.HEIGHT) 
                self.paused = False 
                self.first_run = True # 重置first_run以便重新绘制整个网格
                self.previous_grid_state = [[(cell.trigram, cell.state) for cell in row] for row in self.grid.grid] # Reset previous state as well
        except AttributeError:
            if key == keyboard.Key.esc: 
                self.running = False
                return False

    def on_click(self, x, y, button, pressed):
        """pynput 的鼠标点击事件处理函数。"""
        if pressed and button == mouse.Button.left: 
            # 调整鼠标点击的像素坐标以考虑顶部信息行和终端的字符大小
            # 假设每个字符约 CHAR_WIDTH_PIXELS 宽， CHAR_HEIGHT_PIXELS 高。
            # grid_start_row 是网格内容开始的行号 (1-indexed)
            grid_start_row_in_pixels = (self.header_lines + 2) * CHAR_HEIGHT_PIXELS + GRID_TOP_OFFSET_PIXELS

            row, col = get_grid_coords_from_pixels(
                x, y,
                CHAR_WIDTH_PIXELS, CHAR_HEIGHT_PIXELS,
                GRID_LEFT_OFFSET_PIXELS, grid_start_row_in_pixels, # Adjust top offset based on header lines
                GameConfig.CELL_DISPLAY_WIDTH
            )

            if 0 <= row < self.grid.height and 0 <= col < self.grid.width: 
                current_trigram = self.grid.get_trigram_at(row, col) 
                if current_trigram: 
                    idx = (self.trigrams_cycle.index(current_trigram) + 1) % len(self.trigrams_cycle)
                    self.grid.set_trigram_at(row, col, self.trigrams_cycle[idx]) 

    def run(self):
        """主游戏循环。"""
        try:
            if PYNPUT_SUPPORT: 
                self.keyboard_listener = keyboard.Listener(on_press=self.on_press)
                self.keyboard_listener.start()
                self.mouse_listener = mouse.Listener(on_click=self.on_click)
                self.mouse_listener.start()

            while self.running: 
                self.print_grid() 
                if not self.paused: 
                    self.grid.update_grid() 

                if not PYNPUT_SUPPORT:
                    input()
                else:
                    time.sleep(self.update_interval) 
        except Exception as e:
            print(Fore.RED + f"\n发生意外错误：{e}" + Style.RESET_ALL)
            import traceback
            traceback.print_exc() 
        finally:
            if PYNPUT_SUPPORT:
                if self.keyboard_listener and self.keyboard_listener.is_alive():
                    self.keyboard_listener.stop()
                if self.mouse_listener and self.mouse_listener.is_alive():
                    self.mouse_listener.stop()
            print("\n游戏结束。正在退出。")
            sys.exit(0) 

if __name__ == "__main__":
    game = Game() 
    game.run()