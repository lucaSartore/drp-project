from map.constants import MAP_X_LOWER_BOUND, MAP_X_UPPER_BOUND, MAP_Y_LOWER_BOUND, MAP_Y_UPPER_BOUND
from map.data_type import Point
import pygame 


class Display:
    """Display class to visualize the game state using pygame."""
    
    WINDOW_WIDTH = 1400
    WINDOW_HEIGHT = 700
    PADDING = 20
    
    # Colors
    BACKGROUND_COLOR = (255, 255, 255)
    BORDER_COLOR = (0, 0, 0)
    CHASER_COLOR = (255, 0, 0)
    RUNNER_COLOR = (0, 255, 0)
    FAKE_RUNNER_COLOR = (128, 128, 128)
    VISION_CIRCLE_COLOR = (200, 200, 200)
    
    # Entity sizes
    ENTITY_RADIUS = 5
    VISION_ALPHA = 100

    def __init__(self) -> None:
        """Initialize pygame display with two side-by-side maps."""
        pygame.init()
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("Map Visualization")
        self.clock = pygame.time.Clock()
        
        # Calculate dimensions for each side
        self.side_width = (self.WINDOW_WIDTH - 3 * self.PADDING) // 2
        self.side_height = self.WINDOW_HEIGHT - 2 * self.PADDING
        
        # Left side position
        self.left_x = self.PADDING
        self.left_y = self.PADDING
        
        # Right side position
        self.right_x = self.PADDING * 2 + self.side_width
        self.right_y = self.PADDING

    def _world_to_screen(self, point: Point, side_x: int, side_y: int) -> tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        # Normalize world coordinates to [0, 1]
        norm_x = (point.x - MAP_X_LOWER_BOUND) / (MAP_X_UPPER_BOUND - MAP_X_LOWER_BOUND)
        norm_y = (point.y - MAP_Y_LOWER_BOUND) / (MAP_Y_UPPER_BOUND - MAP_Y_LOWER_BOUND)
        
        # Convert to screen coordinates
        screen_x = int(side_x + norm_x * self.side_width)
        screen_y = int(side_y + norm_y * self.side_height)
        
        return screen_x, screen_y

    def _draw_map_area(self, side_x: int, side_y: int):
        """Draw the map boundary rectangle."""
        pygame.draw.rect(
            self.screen,
            self.BORDER_COLOR,
            (side_x, side_y, self.side_width, self.side_height),
            2
        )

    def _draw_entity(self, point: Point, side_x: int, side_y: int, color: tuple):
        """Draw a circular entity at the given position."""
        screen_x, screen_y = self._world_to_screen(point, side_x, side_y)
        pygame.draw.circle(self.screen, color, (screen_x, screen_y), self.ENTITY_RADIUS)

    def _draw_vision_circle(self, point: Point, radius: float, side_x: int, side_y: int):
        """Draw a transparent vision circle."""
        screen_x, screen_y = self._world_to_screen(point, side_x, side_y)
        
        # Convert world radius to screen radius
        pixels_per_world_unit = self.side_width / (MAP_X_UPPER_BOUND - MAP_X_LOWER_BOUND)
        screen_radius = int(radius * pixels_per_world_unit)
        
        # Create a surface for the transparent circle
        circle_surface = pygame.Surface((screen_radius * 2, screen_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(
            circle_surface,
            (*self.VISION_CIRCLE_COLOR, self.VISION_ALPHA),
            (screen_radius, screen_radius),
            screen_radius
        )
        
        # Blit the circle surface onto the main screen
        self.screen.blit(circle_surface, (screen_x - screen_radius, screen_y - screen_radius))

    def update_left_side(self, settings, chasers: list[Point], runner: Point, fake_runners: list[Point]):
        """
        Draw the left side of the display with:
        - Chasers as red dots with field of vision circles
        - Runner as green dot with field of vision circle
        - Fake runners as gray dots
        """
        # Draw map boundary
        self._draw_map_area(self.left_x, self.left_y)
        
        # Draw chasers with vision circles
        for chaser in chasers:
            self._draw_vision_circle(chaser, settings.chaser_detection_radius, self.left_x, self.left_y)
            self._draw_entity(chaser, self.left_x, self.left_y, self.CHASER_COLOR)
        
        # Draw runner with vision circle
        self._draw_vision_circle(runner, settings.runner_detection_radius, self.left_x, self.left_y)
        self._draw_entity(runner, self.left_x, self.left_y, self.RUNNER_COLOR)
        
        # Draw fake runners
        for fake_runner in fake_runners:
            self._draw_entity(fake_runner, self.left_x, self.left_y, self.FAKE_RUNNER_COLOR)

    def update_right_side(self):
        """Still to implement."""
        # Draw map boundary for right side
        self._draw_map_area(self.right_x, self.right_y)

    def render(self):
        """Update the display."""
        pygame.display.flip()

    def clear(self):
        """Clear the screen."""
        self.screen.fill(self.BACKGROUND_COLOR)

    def close(self):
        """Close the display."""
        pygame.quit()
