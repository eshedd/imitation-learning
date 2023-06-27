import os
import random
import pygame
import torch
import argparse


# Class for the orange dude
class Player(object):

    def __init__(self):
        self.rect = pygame.Rect(32, 32, 16, 16)

# Nice class to hold a germ rect
class Germ(object):
    
    def __init__(self, pos):
        germs.append(self)
        self.rect = pygame.Rect(pos[0], pos[1], 16, 16)
    
    def fall(self):
        self.rect.y += 5


# Initialise pygame
os.environ["SDL_VIDEO_CENTERED"] = "1"  # centers window
pygame.init()

# Set up the display
pygame.display.set_caption("Imitation Learning Test")
screen_x, screen_y = 75, 100
screen = pygame.display.set_mode((screen_x, screen_y))
screen_rect = screen.get_rect()

clock = pygame.time.Clock()
germs = [] # List to hold the germs

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--human', action='store_true')  # human mode
    args = parser.parse_args()

    player = Player() # Create the player
    SPAWNEVENT = pygame.USEREVENT+0
    MOVEEVENT = pygame.USEREVENT+1
    pygame.time.set_timer(SPAWNEVENT, 500)
    pygame.time.set_timer(MOVEEVENT, 50)

    human = args.human
    moves = 'x'
    states = []
    running = True
    while running:

        clock.tick(60)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False
            if e.type == SPAWNEVENT:
                Germ((random.randint(0, screen_x-16), 0))
            if e.type == MOVEEVENT:
                for germ in germs:
                    germ.fall()

        # Move the player if an arrow key is pressed
        if human:
            key = pygame.key.get_pressed()
            if key[pygame.K_LEFT]:
                player.rect.x += -2
                move = 'l'
            elif key[pygame.K_RIGHT]:
                player.rect.x += 2
                move = 'r'
            elif key[pygame.K_UP]:
                player.rect.y += -2
                move = 'u'
            elif key[pygame.K_DOWN]:
                player.rect.y += 2
                move = 'd'
            else:
                move = 's'
        else:
            model = torch.load('imitator.pt')
            model.eval()
            state = [[player.rect.x, player.rect.y]] + [[germ.rect.x, germ.rect.y] for germ in germs]
            state = torch.tensor(state + ([[-100, -100]] * (4 - len(state))))
            state = state[None, :, :]
            move = model(state).argmax().item()
            if move == 0:
                player.rect.x += -2
                move = 'l'
            elif move == 1:
                player.rect.x += 2
                move = 'r'
            elif move == 2:
                player.rect.y += -2
                move = 'u'
            elif move == 3:
                player.rect.y += 2
                move = 'd'
            else:
                move = 's'
        
        player.rect.clamp_ip(screen_rect)

        # Draw the scene
        screen.fill((0, 0, 0))
        state = [[player.rect.x, player.rect.y]]
        for germ in germs:
            pygame.draw.rect(screen, (255, 255, 255), germ.rect)
            if player.rect.colliderect(germ.rect):
                running = False
            if germ.rect.y > screen_y:
                germs.remove(germ)
            else:
                state.append([germ.rect.x, germ.rect.y])
        
        if moves[-1] != move:
            moves += move
            states.append(state)
        pygame.draw.rect(screen, (255, 255, 0), player.rect)
        pygame.display.flip()
    
    if human:
        file_name = 'data.txt'
    else:
        file_name = 'imitation.txt'
    with open(file_name, 'a') as f:
        f.write(moves + ' | ' + str(states) + '\n')

    raise SystemExit