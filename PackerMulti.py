import os
import json
import argparse
import enum
from PIL import Image
from typing import List, Dict, Tuple, Optional


class Rectangle:
    """Represents a rectangle with width, height, and position (x, y)."""
    def __init__(self, width: int, height: int, x: int = 0, y: int = 0, name: str = "", rotated: bool = False):
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.name = name
        self.rotated = rotated  # True if the sprite is rotated 90 degrees

    def __repr__(self):
        rotation = " (rotated)" if self.rotated else ""
        return f"Rectangle({self.width}×{self.height} at ({self.x},{self.y}){rotation} - {self.name})"

    def intersects(self, other: 'Rectangle') -> bool:
        """Check if this rectangle intersects with another."""
        return not (
            self.x + self.width <= other.x or
            self.y + self.height <= other.y or
            self.x >= other.x + other.width or
            self.y >= other.y + other.height
        )

    def area(self) -> int:
        """Get the area of the rectangle."""
        return self.width * self.height


class HeuristicType(enum.Enum):
    """Enum for placement heuristics."""
    BEST_SHORT_SIDE_FIT = 1  # Minimize the shorter leftover side
    BEST_LONG_SIDE_FIT = 2   # Minimize the longer leftover side
    BEST_AREA_FIT = 3        # Minimize the total area of leftover space
    BOTTOM_LEFT = 4          # Place at the bottom-left most position


class SplitHeuristic(enum.Enum):
    """Enum for guillotine split heuristics."""
    SHORTEST_AXIS = 1  # Split along the shortest axis
    LONGEST_AXIS = 2   # Split along the longest axis
    MIN_AREA = 3       # Minimize the area of the resulting rectangles
    MAX_AREA = 4       # Maximize the area of the resulting rectangles


class PackingAlgorithm(enum.Enum):
    """Enum for packing algorithms."""
    MAX_RECTS = 1  # Maximal Rectangles algorithm


class MaxRectsPackerSheet:
    """Implementation of the Maximal Rectangles algorithm for a single sheet."""
    
    def __init__(self, width: int, height: int, padding: int = 1, allow_rotation: bool = False, 
                 heuristic: HeuristicType = HeuristicType.BEST_SHORT_SIDE_FIT):
        self.width = width
        self.height = height
        self.padding = padding
        self.allow_rotation = allow_rotation
        self.heuristic = heuristic
        # Start with the entire sheet as a free rectangle
        self.free_rects = [Rectangle(width, height)]
        self.placed_rects = []
        
    def can_fit(self, width: int, height: int) -> Optional[Tuple[int, int, bool]]:
        """Check if a rectangle of given size can fit on the sheet.
        Returns (x, y, is_rotated) where it fits best, or None if it doesn't fit."""
        padded_width = width + self.padding * 2
        padded_height = height + self.padding * 2
        
        best_score1 = float('inf')
        best_score2 = float('inf')
        best_pos = None
        best_rotated = False
        
        # Try normal orientation first
        for rect in self.free_rects:
            if rect.width >= padded_width and rect.height >= padded_height:
                # Calculate score based on the selected heuristic
                score1, score2 = self._calculate_score(rect.width, rect.height, padded_width, padded_height)
                
                # Update if this is better than the current best
                if ((score1 < best_score1) or 
                    (score1 == best_score1 and score2 < best_score2)):
                    best_score1 = score1
                    best_score2 = score2
                    best_pos = (rect.x, rect.y)
                    best_rotated = False
        
        # Try rotated orientation if allowed
        if self.allow_rotation and width != height:  # No need to check rotation for squares
            rotated_width, rotated_height = padded_height, padded_width
            
            for rect in self.free_rects:
                if rect.width >= rotated_width and rect.height >= rotated_height:
                    # Calculate score based on the selected heuristic
                    score1, score2 = self._calculate_score(rect.width, rect.height, rotated_width, rotated_height)
                    
                    # Update if this is better than the current best
                    if ((score1 < best_score1) or 
                        (score1 == best_score1 and score2 < best_score2)):
                        best_score1 = score1
                        best_score2 = score2
                        best_pos = (rect.x, rect.y)
                        best_rotated = True
                    
        if best_pos is None:
            return None
        
        return (*best_pos, best_rotated)
    
    def _calculate_score(self, free_width: int, free_height: int, width: int, height: int) -> Tuple[float, float]:
        """Calculate the score based on the selected heuristic."""
        leftover_width = free_width - width
        leftover_height = free_height - height
        
        if self.heuristic == HeuristicType.BEST_SHORT_SIDE_FIT:
            # Return score based on short side fit (primary) and long side fit (secondary)
            return min(leftover_width, leftover_height), max(leftover_width, leftover_height)
        
        elif self.heuristic == HeuristicType.BEST_LONG_SIDE_FIT:
            # Return score based on long side fit (primary) and short side fit (secondary)
            return max(leftover_width, leftover_height), min(leftover_width, leftover_height)
        
        elif self.heuristic == HeuristicType.BEST_AREA_FIT:
            # Return score based on area fit (primary) and short side fit (secondary)
            return leftover_width * leftover_height, min(leftover_width, leftover_height)
        
        elif self.heuristic == HeuristicType.BOTTOM_LEFT:
            # Return score based on y-coordinate (primary) and x-coordinate (secondary)
            # Note: we're using the coordinates of the free rectangle here
            return free_height, free_width
        
        # Default to short side fit
        return min(leftover_width, leftover_height), max(leftover_width, leftover_height)
    
    def insert(self, width: int, height: int, name: str) -> Optional[Rectangle]:
        """Try to insert a rectangle with given dimensions. Returns the placed rectangle or None if it couldn't fit."""
        # Check if we can fit the rectangle
        fit_result = self.can_fit(width, height)
        if fit_result is None:
            return None
            
        x, y, is_rotated = fit_result
        
        # If rotated, swap width and height
        actual_width, actual_height = width, height
        if is_rotated:
            actual_width, actual_height = height, width
        
        # Create the rectangle with padding
        placed_rect = Rectangle(actual_width, actual_height, x + self.padding, y + self.padding, name, is_rotated)
        
        # Create the full rectangle including padding for internal calculations
        padded_width = actual_width + self.padding * 2
        padded_height = actual_height + self.padding * 2
        full_rect = Rectangle(padded_width, padded_height, x, y)
        
        # Add to placed rects
        self.placed_rects.append(placed_rect)
        
        # Split free rectangles
        self._split_free_rectangles(full_rect)
        
        # Prune redundant free rectangles
        self._prune_free_rectangles()
        
        return placed_rect
        
    def _split_free_rectangles(self, inserted_rect: Rectangle):
        """Split all free rectangles that overlap with the inserted rectangle."""
        new_free_rects = []
        
        for free_rect in self.free_rects:
            if not inserted_rect.intersects(free_rect):
                new_free_rects.append(free_rect)
                continue
                
            # Calculate the four possible split rectangles
            if inserted_rect.x < free_rect.x + free_rect.width and inserted_rect.x + inserted_rect.width > free_rect.x:
                # New rectangle above the inserted rect
                if inserted_rect.y > free_rect.y:
                    new_rect = Rectangle(
                        free_rect.width,
                        inserted_rect.y - free_rect.y,
                        free_rect.x,
                        free_rect.y
                    )
                    new_free_rects.append(new_rect)
                
                # New rectangle below the inserted rect
                if inserted_rect.y + inserted_rect.height < free_rect.y + free_rect.height:
                    new_rect = Rectangle(
                        free_rect.width,
                        free_rect.y + free_rect.height - (inserted_rect.y + inserted_rect.height),
                        free_rect.x,
                        inserted_rect.y + inserted_rect.height
                    )
                    new_free_rects.append(new_rect)
            
            if inserted_rect.y < free_rect.y + free_rect.height and inserted_rect.y + inserted_rect.height > free_rect.y:
                # New rectangle to the left of the inserted rect
                if inserted_rect.x > free_rect.x:
                    new_rect = Rectangle(
                        inserted_rect.x - free_rect.x,
                        free_rect.height,
                        free_rect.x,
                        free_rect.y
                    )
                    new_free_rects.append(new_rect)
                
                # New rectangle to the right of the inserted rect
                if inserted_rect.x + inserted_rect.width < free_rect.x + free_rect.width:
                    new_rect = Rectangle(
                        free_rect.x + free_rect.width - (inserted_rect.x + inserted_rect.width),
                        free_rect.height,
                        inserted_rect.x + inserted_rect.width,
                        free_rect.y
                    )
                    new_free_rects.append(new_rect)
        
        self.free_rects = new_free_rects
    
    def _prune_free_rectangles(self):
        """Remove redundant free rectangles (those completely contained within others)."""
        # If we have too many free rectangles, do a more aggressive pruning
        if len(self.free_rects) > 1000:
            # Keep only the largest free rectangles
            self.free_rects.sort(key=lambda r: r.width * r.height, reverse=True)
            self.free_rects = self.free_rects[:1000]
            return
            
        i = 0
        while i < len(self.free_rects):
            j = i + 1
            while j < len(self.free_rects):
                # If rect i is contained in rect j
                if (self.free_rects[i].x >= self.free_rects[j].x and
                    self.free_rects[i].y >= self.free_rects[j].y and
                    self.free_rects[i].x + self.free_rects[i].width <= self.free_rects[j].x + self.free_rects[j].width and
                    self.free_rects[i].y + self.free_rects[i].height <= self.free_rects[j].y + self.free_rects[j].height):
                    # Remove rect i
                    self.free_rects.pop(i)
                    i -= 1
                    break
                # If rect j is contained in rect i
                elif (self.free_rects[j].x >= self.free_rects[i].x and
                      self.free_rects[j].y >= self.free_rects[i].y and
                      self.free_rects[j].x + self.free_rects[j].width <= self.free_rects[i].x + self.free_rects[i].width and
                      self.free_rects[j].y + self.free_rects[j].height <= self.free_rects[i].y + self.free_rects[i].height):
                    # Remove rect j
                    self.free_rects.pop(j)
                else:
                    j += 1
            i += 1


def next_power_of_two(n: int) -> int:
    """Return the next power of two greater than or equal to n."""
    if n <= 0:
        return 1
    # Decrement n (to handle cases when n is already a power of 2)
    n -= 1
    # Set all bits after the most significant bit
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    # Increment to get next power of 2
    return n + 1


def trim_whitespace(img: Image.Image) -> Tuple[Image.Image, int, int, int, int]:
    """
    Trim transparent whitespace from an image.
    Returns the trimmed image and the offsets (left, top, width, height).
    """
    # Get alpha channel if available, otherwise assume fully opaque
    if img.mode == 'RGBA':
        alpha = img.split()[3]
    else:
        # If no alpha, convert to RGBA and assume fully opaque
        img = img.convert('RGBA')
        alpha = Image.new('L', img.size, 255)
    
    # Get bounding box of non-zero regions
    bbox = alpha.getbbox()
    
    # If bbox is None (entirely transparent image), return original
    if bbox is None:
        return img, 0, 0, img.width, img.height
    
    # Calculate offsets
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    
    # Crop image to bbox
    trimmed_img = img.crop(bbox)
    
    return trimmed_img, left, top, width, height


class SpriteSheetPacker:
    """Packs multiple sprites into power-of-two sized sprite sheets."""
    
    def __init__(self, max_width: int = 4096, max_height: int = 4096, padding: int = 1, 
                 allow_rotation: bool = False, trim_sprites: bool = False,
                 placement_heuristic: HeuristicType = HeuristicType.BEST_SHORT_SIDE_FIT):
        self.max_width = max_width
        self.max_height = max_height
        self.padding = padding
        self.allow_rotation = allow_rotation
        self.trim_sprites = trim_sprites
        self.placement_heuristic = placement_heuristic
        self.sheets = []
        self.current_sheet = None
        self.current_sheet_size = (256, 256)  # Start with a small sheet and grow as needed
    
    def _create_new_sheet(self):
        """Create a new sheet with the current size."""
        width, height = self.current_sheet_size
        self.current_sheet = MaxRectsPackerSheet(
            width, height, self.padding, self.allow_rotation, self.placement_heuristic
        )
        self.sheets.append(self.current_sheet)
    
    def _grow_sheet_size(self):
        """Double the sheet size in the shorter dimension, up to max limits."""
        width, height = self.current_sheet_size
        
        if width <= height and width < self.max_width:
            width = min(width * 2, self.max_width)
        elif height < self.max_height:
            height = min(height * 2, self.max_height)
        else:
            # Can't grow any more, maximum size reached
            return False
        
        self.current_sheet_size = (width, height)
        return True
    
    def pack_sprites(self, sprite_paths: List[str]) -> List[Tuple[List[Dict], Image.Image]]:
        """Pack sprites into sheets and return atlas data and sheet images."""
        
        # Load all sprites and sort them by area (largest first)
        sprites = []
        trim_data = {}  # Store trim data for each sprite
        
        for path in sprite_paths:
            try:
                img = Image.open(path)
                name = os.path.basename(path)
                
                # Trim whitespace if enabled
                if self.trim_sprites:
                    trimmed_img, trim_x, trim_y, trim_width, trim_height = trim_whitespace(img)
                    # Store trim data for later use
                    trim_data[name] = {
                        "original_width": img.width,
                        "original_height": img.height,
                        "trim_x": trim_x,
                        "trim_y": trim_y,
                        "trimmed_width": trim_width,
                        "trimmed_height": trim_height
                    }
                    # Use the trimmed image for packing
                    sprites.append((name, trimmed_img, trimmed_img.width, trimmed_img.height))
                else:
                    sprites.append((name, img, img.width, img.height))
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        # Sort by area (largest first)
        sprites.sort(key=lambda s: s[2] * s[3], reverse=True)
        
        # Print info about the sprites
        print(f"Packing {len(sprites)} sprites")
        total_area = sum(w * h for _, _, w, h in sprites)
        print(f"Total sprite area: {total_area} pixels")
        
        # Estimate required area with some wastage
        estimated_area = total_area * 1.5  # 50% extra for wastage
        sheet_width = min(self.max_width, next_power_of_two(int(estimated_area ** 0.5)))
        sheet_height = min(self.max_height, next_power_of_two(int(estimated_area ** 0.5)))
        print(f"Starting with sheet size: {sheet_width}×{sheet_height}")
        
        # Start with a reasonably sized sheet
        self.current_sheet_size = (sheet_width, sheet_height)
        
        results = []
        remaining_sprites = sprites.copy()
        sheet_count = 0
        max_sheets = 10  # Safety limit to prevent infinite loops
        
        while remaining_sprites and sheet_count < max_sheets:
            sheet_count += 1
            print(f"Processing sheet {sheet_count} with {len(remaining_sprites)} sprites remaining")
            
            # Create a new sheet with current size
            self._create_new_sheet()
            
            packed_sprites = []
            sprites_to_retry = []
            
            # Try to pack sprites into current sheet
            retry_count = 0
            growth_count = 0
            max_growth = 3  # Limit sheet growth attempts
            
            while remaining_sprites and growth_count < max_growth:
                name, img, width, height = remaining_sprites.pop(0)
                
                # Skip sprites that are too large for max sheet size
                if width > self.max_width or height > self.max_height:
                    print(f"Sprite {name} ({width}×{height}) exceeds maximum sheet dimensions and will be skipped")
                    continue
                
                # Try to insert the sprite
                placed_rect = self.current_sheet.insert(width, height, name)
                
                if placed_rect is None:
                    # Couldn't fit, see if we can grow the sheet
                    if self._grow_sheet_size():
                        # Sheet grew, retry with new sheet of increased size
                        print(f"Growing sheet to {self.current_sheet_size[0]}×{self.current_sheet_size[1]}")
                        growth_count += 1
                        
                        # Reset and try again with new sheet
                        remaining_sprites.insert(0, (name, img, width, height))
                        remaining_sprites = packed_sprites + remaining_sprites
                        packed_sprites = []
                        self._create_new_sheet()  # Create new sheet with new size
                    else:
                        # Can't grow more, save for next sheet
                        sprites_to_retry.append((name, img, width, height))
                else:
                    # Successfully placed
                    packed_sprites.append((placed_rect, img))
                    
                # Avoid infinite loops with sprites that can't be placed
                retry_count += 1
                if retry_count > len(sprites) * 2:
                    print("Warning: Too many retries, may be stuck in a loop")
                    break
            
            # Create the sheet image from packed sprites
            if packed_sprites:
                print(f"Placed {len(packed_sprites)} sprites on sheet {sheet_count}")
                
                # Find the actual used dimensions
                max_x = max(rect.x + rect.width for rect, _ in packed_sprites)
                max_y = max(rect.y + rect.height for rect, _ in packed_sprites)
                
                # Ensure dimensions are power of two
                pot_width = next_power_of_two(max_x)
                pot_height = next_power_of_two(max_y)
                
                print(f"Sheet {sheet_count} dimensions: {pot_width}×{pot_height}")
                
                # Create the sheet image
                sheet_img = Image.new('RGBA', (pot_width, pot_height), (0, 0, 0, 0))
                
                # Create atlas data
                atlas_data = []
                
                for rect, img in packed_sprites:
                    # If sprite is rotated, rotate the image before pasting
                    if rect.rotated:
                        # Rotate 90 degrees counter-clockwise
                        rotated_img = img.transpose(Image.ROTATE_90)
                        sheet_img.paste(rotated_img, (rect.x, rect.y))
                    else:
                        # Paste normally
                        sheet_img.paste(img, (rect.x, rect.y))
                    
                    # Get trim data if available
                    sprite_trim_data = trim_data.get(rect.name, None) if self.trim_sprites else None
                    
                    # Add to atlas with trim data if available
                    atlas_entry = {
                        "name": rect.name,
                        "x": rect.x,
                        "y": rect.y,
                        "width": rect.width,
                        "height": rect.height,
                        "rotated": rect.rotated
                    }
                    
                    # If trimmed, add original dimensions and offsets
                    if sprite_trim_data:
                        original_width = sprite_trim_data["original_width"]
                        original_height = sprite_trim_data["original_height"]
                        trim_x = sprite_trim_data["trim_x"]
                        trim_y = sprite_trim_data["trim_y"]
                        
                        # Adjust for rotation if needed
                        if rect.rotated:
                            # For rotated sprites, we need to adjust the offset differently
                            atlas_entry["trimmed"] = True
                            atlas_entry["spriteSourceSize"] = {
                                "x": trim_y,
                                "y": original_width - trim_x - rect.width,
                                "w": rect.width,
                                "h": rect.height
                            }
                            atlas_entry["sourceSize"] = {
                                "w": original_height,
                                "h": original_width
                            }
                        else:
                            atlas_entry["trimmed"] = True
                            atlas_entry["spriteSourceSize"] = {
                                "x": trim_x,
                                "y": trim_y,
                                "w": rect.width,
                                "h": rect.height
                            }
                            atlas_entry["sourceSize"] = {
                                "w": original_width,
                                "h": original_height
                            }
                    else:
                        # If not trimmed, use defaults
                        atlas_entry["trimmed"] = False
                        atlas_entry["spriteSourceSize"] = {
                            "x": 0,
                            "y": 0,
                            "w": rect.width,
                            "h": rect.height
                        }
                        atlas_entry["sourceSize"] = {
                            "w": rect.width if not rect.rotated else rect.height,
                            "h": rect.height if not rect.rotated else rect.width
                        }
                    
                    atlas_data.append(atlas_entry)
                
                results.append((atlas_data, sheet_img))
            
            # Reset for next sheet
            self.current_sheet = None
            remaining_sprites = sprites_to_retry
            
        return results


def main():
    parser = argparse.ArgumentParser(description='Pack sprite frames into optimized sprite sheets')
    parser.add_argument('input_dir', help='Directory containing sprite frames')
    parser.add_argument('output_dir', help='Directory to save sprite sheets and atlases')
    parser.add_argument('--max-width', type=int, default=4096, help='Maximum width of sprite sheets')
    parser.add_argument('--max-height', type=int, default=4096, help='Maximum height of sprite sheets')
    parser.add_argument('--padding', type=int, default=1, help='Padding between sprites')
    parser.add_argument('--prefix', default='sheet', help='Prefix for output files')
    parser.add_argument('--rotate', action='store_true', help='Allow rotation of sprites by 90 degrees')
    parser.add_argument('--trim', action='store_true', help='Trim transparent borders from sprites')
    parser.add_argument('--placement', type=str, default='shortside', 
                        choices=['shortside', 'longside', 'area', 'bottomleft', 'optimal'],
                        help='Placement heuristic to use (use "optimal" to try all and pick best)')
    
    args = parser.parse_args()
    
    # Map string arguments to enum values
    placement_map = {
        'shortside': HeuristicType.BEST_SHORT_SIDE_FIT,
        'longside': HeuristicType.BEST_LONG_SIDE_FIT,
        'area': HeuristicType.BEST_AREA_FIT,
        'bottomleft': HeuristicType.BOTTOM_LEFT
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all sprite files
    sprite_files = []
    print(f"Scanning directory: {args.input_dir}")
    
    # Use os.listdir instead of walk to avoid recursion
    try:
        files = os.listdir(args.input_dir)
        for file in files:
            full_path = os.path.join(args.input_dir, file)
            if os.path.isfile(full_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                sprite_files.append(full_path)
    except Exception as e:
        print(f"Error scanning directory: {e}")
        return
    
    if not sprite_files:
        print(f"No sprite files found in {args.input_dir}")
        return
    
    print(f"Found {len(sprite_files)} sprite files")
    
    # Check if we should try all placement heuristics to find the optimal one
    if args.placement.lower() == 'optimal':
        print("Finding optimal placement heuristic...")
        best_efficiency = 0
        best_heuristic = None
        best_results = None
        
        # Try each placement heuristic
        for heuristic_name, heuristic_value in placement_map.items():
            print(f"Trying {heuristic_name} placement...")
            
            # Create packer with current heuristic
            packer = SpriteSheetPacker(
                args.max_width,
                args.max_height,
                args.padding,
                args.rotate,
                args.trim,
                heuristic_value
            )
            
            # Pack sprites
            results = packer.pack_sprites(sprite_files)
            
            # Calculate efficiency
            total_pixels = 0
            sprite_pixels = 0
            for atlas_data, sheet_img in results:
                total_pixels += sheet_img.width * sheet_img.height
                sprite_pixels += sum(item["width"] * item["height"] for item in atlas_data)
            
            current_efficiency = (sprite_pixels / total_pixels) * 100 if total_pixels > 0 else 0
            print(f"  {heuristic_name} efficiency: {current_efficiency:.2f}%")
            
            # Update best if this is more efficient
            if current_efficiency > best_efficiency:
                best_efficiency = current_efficiency
                best_heuristic = heuristic_name
                best_results = results
        
        print(f"\nBest placement heuristic: {best_heuristic} with {best_efficiency:.2f}% efficiency")
        results = best_results
        placement_heuristic_name = best_heuristic
    else:
        # Use the specified placement heuristic
        placement_heuristic = placement_map.get(args.placement.lower(), HeuristicType.BEST_SHORT_SIDE_FIT)
        
        # Create packer with the specified dimensions and settings
        packer = SpriteSheetPacker(
            args.max_width,
            args.max_height,
            args.padding,
            args.rotate,
            args.trim,
            placement_heuristic
        )
        
        # Use the packer to process sprites
        print(f"Using {args.placement} placement heuristic")
        results = packer.pack_sprites(sprite_files)
        placement_heuristic_name = args.placement
    
    # Save results
    print("Saving results...")
    for i, (atlas_data, sheet_img) in enumerate(results):
        sheet_path = os.path.join(args.output_dir, f"{args.prefix}_{i}.png")
        atlas_path = os.path.join(args.output_dir, f"{args.prefix}_{i}.json")
        
        print(f"Saving sheet {i+1}/{len(results)}: {sheet_path} ({sheet_img.width}×{sheet_img.height}) with {len(atlas_data)} sprites")
        
        # Save sheet
        sheet_img.save(sheet_path)
        
        # Save atlas with the enhanced data
        atlas = {
            "frames": {item["name"]: {
                "frame": {"x": item["x"], "y": item["y"], "w": item["width"], "h": item["height"]},
                "rotated": item["rotated"],
                "trimmed": item["trimmed"],
                "spriteSourceSize": item["spriteSourceSize"],
                "sourceSize": item["sourceSize"]
            } for item in atlas_data},
            "meta": {
                "image": os.path.basename(sheet_path),
                "format": "RGBA8888",
                "size": {"w": sheet_img.width, "h": sheet_img.height},
                "scale": "1",
                "placement_heuristic": placement_heuristic_name
            }
        }
        
        with open(atlas_path, 'w') as f:
            json.dump(atlas, f, indent=2)
    
    # Calculate and print efficiency
    total_pixels = 0
    sprite_pixels = 0
    for atlas_data, sheet_img in results:
        total_pixels += sheet_img.width * sheet_img.height
        sprite_pixels += sum(item["width"] * item["height"] for item in atlas_data)
    
    efficiency = (sprite_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    print(f"\nFinal packing efficiency: {efficiency:.2f}%")
    print("Done!")


if __name__ == "__main__":
    main()
