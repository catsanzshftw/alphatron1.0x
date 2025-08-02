import random
import math
from ursina import *
from ursina.prefabs.first_person_controller import FirstPersonController
from perlin_noise import PerlinNoise
import numpy as np

# --- Initialize Application ---
app = Ursina()
window.borderless = False
window.fullscreen = False
window.exit_button.visible = False
window.fps_counter.enabled = True
application.target_fps = 60

# --- Constants ---
CHUNK_SIZE = 16
RENDER_DISTANCE = 4
WORLD_HEIGHT = 128
SEA_LEVEL = 62
BEDROCK_LEVEL = 5

# --- Block Definitions ---
class BlockType:
    AIR = 0
    GRASS = 1
    DIRT = 2
    STONE = 3
    COBBLESTONE = 4
    WOOD = 5
    LEAVES = 6
    SAND = 7
    WATER = 8
    BEDROCK = 9
    COAL_ORE = 10
    IRON_ORE = 11
    GOLD_ORE = 12
    DIAMOND_ORE = 13
    PLANKS = 14
    GRAVEL = 15

BLOCK_COLORS = {
    BlockType.GRASS: color.rgb(95, 159, 53),
    BlockType.DIRT: color.rgb(134, 96, 67),
    BlockType.STONE: color.rgb(136, 136, 136),
    BlockType.COBBLESTONE: color.rgb(115, 115, 115),
    BlockType.WOOD: color.rgb(103, 82, 49),
    BlockType.LEAVES: color.rgb(54, 135, 40),
    BlockType.SAND: color.rgb(219, 207, 163),
    BlockType.WATER: color.rgba(64, 164, 223, 150),
    BlockType.BEDROCK: color.rgb(33, 33, 33),
    BlockType.COAL_ORE: color.rgb(80, 80, 80),
    BlockType.IRON_ORE: color.rgb(175, 142, 119),
    BlockType.GOLD_ORE: color.rgb(246, 232, 89),
    BlockType.DIAMOND_ORE: color.rgb(98, 219, 214),
    BlockType.PLANKS: color.rgb(156, 127, 78),
    BlockType.GRAVEL: color.rgb(131, 127, 127)
}

BLOCK_NAMES = {
    BlockType.GRASS: "Grass",
    BlockType.DIRT: "Dirt",
    BlockType.STONE: "Stone",
    BlockType.COBBLESTONE: "Cobblestone",
    BlockType.WOOD: "Wood",
    BlockType.LEAVES: "Leaves",
    BlockType.SAND: "Sand",
    BlockType.WATER: "Water",
    BlockType.BEDROCK: "Bedrock",
    BlockType.COAL_ORE: "Coal Ore",
    BlockType.IRON_ORE: "Iron Ore",
    BlockType.GOLD_ORE: "Gold Ore",
    BlockType.DIAMOND_ORE: "Diamond Ore",
    BlockType.PLANKS: "Planks",
    BlockType.GRAVEL: "Gravel"
}

# --- Voxel System ---
class Voxel(Button):
    def __init__(self, position=(0,0,0), block_type=BlockType.GRASS):
        super().__init__(
            parent=scene,
            position=position,
            model='cube',
            origin_y=0.5,
            texture='white_cube',
            color=BLOCK_COLORS.get(block_type, color.white),
            scale=1,
            highlight_color=color.white
        )
        self.block_type = block_type
        
        # Add transparency for water
        if block_type == BlockType.WATER:
            self.alpha = 0.7
    
    def input(self, key):
        if self.hovered:
            if key == 'left mouse down':
                # Break block
                if self.block_type != BlockType.BEDROCK:
                    # Drop appropriate block when broken
                    if self.block_type == BlockType.STONE:
                        dropped_type = BlockType.COBBLESTONE
                    elif self.block_type == BlockType.GRASS:
                        dropped_type = BlockType.DIRT
                    else:
                        dropped_type = self.block_type
                    
                    player.add_to_inventory(dropped_type)
                    world.remove_block(self.position)
                    destroy(self)
            
            elif key == 'right mouse down':
                # Place block
                if player.selected_slot < len(player.inventory) and player.inventory[player.selected_slot]['count'] > 0:
                    new_pos = self.position + mouse.normal
                    if not world.get_block(new_pos):
                        block_type = player.inventory[player.selected_slot]['type']
                        world.add_block(new_pos, block_type)
                        player.inventory[player.selected_slot]['count'] -= 1

# --- World Generation ---
class World:
    def __init__(self):
        self.blocks = {}
        self.chunks = {}
        
        # Noise generators
        self.terrain_noise = PerlinNoise(octaves=4, seed=random.randint(0, 1000000))
        self.cave_noise = PerlinNoise(octaves=3, seed=random.randint(0, 1000000))
        self.ore_noise = PerlinNoise(octaves=2, seed=random.randint(0, 1000000))
        
    def generate_chunk(self, chunk_x, chunk_z):
        chunk_key = (chunk_x, chunk_z)
        if chunk_key in self.chunks:
            return
        
        self.chunks[chunk_key] = True
        
        for x in range(chunk_x * CHUNK_SIZE, (chunk_x + 1) * CHUNK_SIZE):
            for z in range(chunk_z * CHUNK_SIZE, (chunk_z + 1) * CHUNK_SIZE):
                # Generate height
                height = int(self.terrain_noise([x * 0.02, z * 0.02]) * 15 + SEA_LEVEL)
                
                # Generate column
                for y in range(WORLD_HEIGHT):
                    if y == 0:
                        # Bedrock at bottom
                        self.add_block((x, y, z), BlockType.BEDROCK)
                    elif y < BEDROCK_LEVEL and random.random() < 0.3:
                        # Random bedrock near bottom
                        self.add_block((x, y, z), BlockType.BEDROCK)
                    elif y < height - 4:
                        # Stone layer with ores
                        block_type = BlockType.STONE
                        
                        # Cave generation
                        cave_value = self.cave_noise([x * 0.05, y * 0.05, z * 0.05])
                        if cave_value > 0.7 and y < height - 5:
                            continue  # Empty space for cave
                        
                        # Ore generation
                        if y < 16 and random.random() < 0.0005:  # Diamond (rare, deep)
                            block_type = BlockType.DIAMOND_ORE
                        elif y < 32 and random.random() < 0.001:  # Gold (rare)
                            block_type = BlockType.GOLD_ORE
                        elif y < 64 and random.random() < 0.005:  # Iron (common)
                            block_type = BlockType.IRON_ORE
                        elif random.random() < 0.01:  # Coal (very common)
                            block_type = BlockType.COAL_ORE
                        
                        self.add_block((x, y, z), block_type)
                    elif y < height:
                        # Dirt layer
                        self.add_block((x, y, z), BlockType.DIRT)
                    elif y == height:
                        # Surface block
                        if height < SEA_LEVEL - 2:
                            self.add_block((x, y, z), BlockType.SAND)
                        else:
                            self.add_block((x, y, z), BlockType.GRASS)
                    elif y <= SEA_LEVEL:
                        # Water at sea level
                        self.add_block((x, y, z), BlockType.WATER)
                
                # Tree generation
                if height > SEA_LEVEL + 1 and random.random() < 0.005:
                    self.generate_tree(x, height + 1, z)
    
    def generate_tree(self, x, y, z):
        # Tree trunk
        trunk_height = random.randint(4, 6)
        for i in range(trunk_height):
            self.add_block((x, y + i, z), BlockType.WOOD)
        
        # Leaves
        leaf_start = y + trunk_height - 2
        leaf_levels = [
            (2, 2),  # Bottom level - 5x5
            (2, 2),  # Middle level - 5x5
            (1, 1),  # Upper level - 3x3
            (1, 1),  # Top level - 3x3
        ]
        
        for level, (w, d) in enumerate(leaf_levels):
            leaf_y = leaf_start + level
            for dx in range(-w, w + 1):
                for dz in range(-d, d + 1):
                    if dx == 0 and dz == 0 and level < 2:
                        continue  # Skip trunk position
                    if abs(dx) == w and abs(dz) == d and random.random() < 0.3:
                        continue  # Random corners missing
                    self.add_block((x + dx, leaf_y, z + dz), BlockType.LEAVES)
    
    def add_block(self, position, block_type):
        self.blocks[position] = block_type
        Voxel(position=position, block_type=block_type)
    
    def remove_block(self, position):
        if position in self.blocks:
            del self.blocks[position]
    
    def get_block(self, position):
        return self.blocks.get(position)

# --- Player Controller ---
class MinecraftPlayer(FirstPersonController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.speed = 4.5
        self.mouse_sensitivity = Vec2(40, 40)
        self.jump_height = 1.5
        self.gravity = 0.5
        
        # Inventory (hotbar)
        self.inventory = [
            {'type': BlockType.DIRT, 'count': 64},
            {'type': BlockType.STONE, 'count': 64},
            {'type': BlockType.WOOD, 'count': 64},
            {'type': BlockType.PLANKS, 'count': 64},
            {'type': BlockType.COBBLESTONE, 'count': 64},
            {'type': BlockType.SAND, 'count': 64},
            {'type': BlockType.GLASS, 'count': 0},
            {'type': BlockType.COAL_ORE, 'count': 0},
            {'type': BlockType.IRON_ORE, 'count': 0},
        ]
        self.selected_slot = 0
        
        # UI
        self.crosshair = Entity(
            parent=camera.ui,
            model='quad',
            color=color.white,
            scale=0.008,
            rotation_z=45
        )
        
        # Hotbar UI
        self.hotbar_bg = Entity(
            parent=camera.ui,
            model='quad',
            color=color.dark_gray,
            scale=(0.9, 0.1, 1),
            position=(0, -0.45, 0)
        )
        
        self.hotbar_slots = []
        self.slot_texts = []
        for i in range(9):
            slot = Entity(
                parent=camera.ui,
                model='quad',
                color=color.gray if i != self.selected_slot else color.white,
                scale=0.08,
                position=(-0.36 + i * 0.09, -0.45, -0.1)
            )
            self.hotbar_slots.append(slot)
            
            # Slot number text
            slot_num = Text(
                str(i + 1),
                parent=camera.ui,
                position=(-0.36 + i * 0.09, -0.41, -0.2),
                scale=0.7,
                origin=(0, 0)
            )
            
            # Item count text
            count_text = Text(
                '',
                parent=camera.ui,
                position=(-0.34 + i * 0.09, -0.48, -0.2),
                scale=0.5,
                origin=(0, 0)
            )
            self.slot_texts.append(count_text)
        
        self.update_hotbar_display()
        
        # Selected block text
        self.block_text = Text(
            'Selected: Dirt',
            position=(0, 0.45),
            origin=(0, 0),
            scale=2
        )
    
    def update_hotbar_display(self):
        for i in range(9):
            self.hotbar_slots[i].color = color.gray if i != self.selected_slot else color.white
            if i < len(self.inventory) and self.inventory[i]['count'] > 0:
                self.slot_texts[i].text = str(self.inventory[i]['count'])
            else:
                self.slot_texts[i].text = ''
        
        if self.selected_slot < len(self.inventory):
            block_type = self.inventory[self.selected_slot]['type']
            self.block_text.text = f'Selected: {BLOCK_NAMES.get(block_type, "Unknown")}'
    
    def add_to_inventory(self, block_type):
        # Try to add to existing stack
        for slot in self.inventory:
            if slot['type'] == block_type and slot['count'] < 64:
                slot['count'] += 1
                self.update_hotbar_display()
                return
        
        # Find empty slot
        for i, slot in enumerate(self.inventory):
            if slot['count'] == 0:
                slot['type'] = block_type
                slot['count'] = 1
                self.update_hotbar_display()
                return
    
    def input(self, key):
        super().input(key)
        
        # Hotbar selection
        if key in '123456789':
            self.selected_slot = int(key) - 1
            self.update_hotbar_display()
        
        # Sprint
        if key == 'left shift':
            self.speed = 7
        elif key == 'left shift up':
            self.speed = 4.5

# --- Sky and Lighting ---
class MinecraftSky(Entity):
    def __init__(self):
        super().__init__(
            parent=scene,
            model='sphere',
            texture='white_cube',
            scale=500,
            double_sided=True
        )
        self.sun_angle = 0
        self.day_cycle_speed = 0.5  # Degrees per second
        
        # Sun
        self.sun = DirectionalLight()
        self.sun.look_at(Vec3(1, -1, 1))
        
        # Ambient light
        self.ambient = AmbientLight(color=color.rgba(100, 100, 100, 0.2))
    
    def update(self):
        # Day/night cycle
        self.sun_angle += self.day_cycle_speed * time.dt
        if self.sun_angle > 360:
            self.sun_angle -= 360
        
        # Calculate sun position
        sun_height = math.sin(math.radians(self.sun_angle))
        sun_forward = math.cos(math.radians(self.sun_angle))
        
        self.sun.look_at(Vec3(sun_forward, -sun_height, 0))
        
        # Sky color based on sun position
        if sun_height > 0:  # Day
            sky_color = lerp(color.rgb(135, 206, 235), color.rgb(255, 200, 150), max(0, 1 - sun_height))
            self.sun.color = color.white
        else:  # Night
            sky_color = color.rgb(20, 24, 82)
            self.sun.color = color.rgb(50, 50, 50)
        
        self.color = sky_color

# --- Main Game Setup ---
print("Initializing Minecraft 1.0 Clone...")

# Create world
world = World()

# Generate initial chunks around spawn
print("Generating world...")
for cx in range(-RENDER_DISTANCE, RENDER_DISTANCE + 1):
    for cz in range(-RENDER_DISTANCE, RENDER_DISTANCE + 1):
        world.generate_chunk(cx, cz)

# Create player
player = MinecraftPlayer(position=(0, SEA_LEVEL + 10, 0))

# Create sky
sky = MinecraftSky()

# Game instructions
instructions = Text(
    'Left Click: Break | Right Click: Place | 1-9: Select | Shift: Sprint | Space: Jump',
    position=window.bottom,
    origin=(0, -1),
    scale=1.5,
    background=True
)

def update():
    # Generate chunks around player
    player_chunk_x = int(player.position.x // CHUNK_SIZE)
    player_chunk_z = int(player.position.z // CHUNK_SIZE)
    
    for cx in range(player_chunk_x - RENDER_DISTANCE, player_chunk_x + RENDER_DISTANCE + 1):
        for cz in range(player_chunk_z - RENDER_DISTANCE, player_chunk_z + RENDER_DISTANCE + 1):
            if (cx, cz) not in world.chunks:
                world.generate_chunk(cx, cz)

# --- Run Game ---
app.run()
