import pygame 
from pygame import gfxdraw

class Window():
    def __init__(self, size) -> None:
        self.size = size 

        pygame.init()
        self.screen = pygame.display.set_mode((self.size, self.size))
        self.screen.fill((255,255,255))
        pygame.display.flip() 
        running = True

        while running : 
            mouse = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            pygame.display.flip() 

        pygame.quit()

        