import os, json, random
import pygame
import torch
import argparse
import matplotlib.pyplot as plt
from algo import Imitator, Reinforcer


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

def run(model, player_type, dual):

    player = Player()  # Create the player
    clock = pygame.time.Clock()

    actions, states = [], []
    score = 0
    running = True
    
    while running:

        clock.tick(60)

        running = handle_events()

        state = [[player.rect.x, player.rect.y]] + [[germ.rect.x, germ.rect.y] for germ in germs]
        action, state = move_player(player, model, player_type, state)
        
        # Save state and action
        if player_type != 0 or action != 4 or (action == 4 and torch.rand(1).item() < 0.15):
            actions.append(action)
            states.append(state)            
        score += 1

        # Draw the scene
        screen.fill((0, 0, 0))
        
        for germ in germs:
            pygame.draw.rect(screen, (255, 255, 255), germ.rect)
            if player.rect.colliderect(germ.rect):
                running = False
            if germ.rect.y > screen_y:
                germs.remove(germ)

        pygame.draw.rect(screen, (255, 255, 0), player.rect)
        pygame.display.flip()

    # Save data
    if not dual:
        save_data(player_type, actions, states)

    return score

def handle_events():
    running = True
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
    return running

def move_player(player, model, player_style, state):
    # Move the player if an arrow key is pressed
    if player_style == 0:  # human
        key = pygame.key.get_pressed()
        if key[pygame.K_LEFT]:
            player.rect.x += -2
            action = 0
        elif key[pygame.K_RIGHT]:
            player.rect.x += 2
            action = 1
        elif key[pygame.K_UP]:
            player.rect.y += -2
            action = 2
        elif key[pygame.K_DOWN]:
            player.rect.y += 2
            action = 3
        else:
            action = 4
    else:  # model
        y_pred = model(torch.tensor(state + ([[-100, -100]] * (5 - len(state))))[None, :, :])
        action = torch.multinomial(y_pred, 1)  # sample from distribution
        action = action.item()
        if action == 0:
            player.rect.x += -2
        elif action == 1:
            player.rect.x += 2
        elif action == 2:
            player.rect.y += -2
        elif action == 3:
            player.rect.y += 2
        else:
            pass
    player.rect.clamp_ip(screen_rect)

    return action, state

def save_data(player_type, actions, states):
    if player_type == 0:
        file_name = './data/human.json'
    elif player_type == 1:
        file_name = './data/imitation.json'
    elif player_type == 2:
        file_name = './data/reinforcement.json'
    else:
        print('Invalid player_type:', player_type)
    
    if not os.path.exists(file_name):
        with open(file_name, 'w') as f:
            json.dump({'episodes': []}, f)

    with open(file_name, 'r') as f:
        dic = json.load(f)
    dic['episodes'].append({'actions': actions, 'states': states})
    with open(file_name, 'w') as f:
        json.dump(dic, f)

def reinforce(model, optimizer, gamma=0.99):
    model.train()
    file_name = './data/reinforcement.json'
    with open(file_name) as f:
        episodes = json.load(f)['episodes']

    states, actions, q_values = [], [], []
    for episode in episodes:
        # get q values
        q_vals = torch.tensor([1] * len(episode['states']))
        n = len(q_vals)
        q_vals = torch.tensor([torch.sum(torch.mul(q_vals[i:], torch.pow(gamma, torch.range(start=i, end=n-1)))) for i in range(n)])
        if len(q_vals) < 11:
            q_vals *= -1
        else:
            for i in range(1, 11):
                q_vals[-i] = -10 + i
        print(q_vals)
        q_values.extend(q_vals)
        # get acts
        acts = torch.tensor(episode['actions'])
        acts = [torch.nn.functional.one_hot(action, num_classes=5) for action in acts]
        acts = torch.stack(acts)
        actions.append(acts)
        # get states
        s = episode['states']
        s = [state + ([[-100, -100]] * (5 - len(state))) for state in s]
        states.append(torch.tensor(s))

    states, actions, q_values = torch.cat(states), torch.cat(actions), torch.tensor(q_values)
    y_pred = model(states)
    negative_likelihoods = -torch.nn.functional.cross_entropy(y_pred.float(), actions.float())
    loss = torch.mul(negative_likelihoods, q_values).mean()
    loss.backward()
    optimizer.step()
    
    # clean up file for next reinforcement policy
    with open(file_name, 'w') as f:
            json.dump({'episodes': []}, f)

    return loss.item()

def save_graph(losses):
    plt.plot(losses, label='loss')
    plt.legend()
    plt.xlabel('update')
    plt.ylabel('loss')
    plt.savefig('./model/r_loss.png')
    print('best loss:', min(losses))

# Initialize pygame
os.environ["SDL_VIDEO_CENTERED"] = "1"  # centers window
pygame.init()

# Set up the display
pygame.display.set_caption("Imitation Learning Test")
screen_x, screen_y = 75, 100
screen = pygame.display.set_mode((screen_x, screen_y))
screen_rect = screen.get_rect()

SPAWNEVENT = pygame.USEREVENT+0
MOVEEVENT = pygame.USEREVENT+1
pygame.time.set_timer(SPAWNEVENT, 500)
pygame.time.set_timer(MOVEEVENT, 50)

germs = []

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--player')  # player mode
    parser.add_argument('-d', '--dual', action='store_true')  # dual mode
    args = parser.parse_args()

    player_type = int(args.player)
    print('Player type:', ['human', 'imitation', 'reinforcement'][player_type])
    dual = args.dual
    
    # model = torch.load('./model/reinforced.pt')  # load up, brother
    model = Reinforcer()
    model.eval()

    if dual:
        random.seed(1)
        bot_score = run(model, player_type=1, dual=True)
        germs = []
        random.seed(1)
        score = run(model, player_type=player_type, dual=True)
        print('Bot score:', bot_score)
        print('Human score:', score)

    elif player_type == 2:  # reinforcement
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        updates = 1000  # number of times to update the model
        episode_count = 1  # number of episodes to generate per update
        losses = []
        for update in range(updates):
            print('Update:', update)
            # generate samples
            with torch.no_grad():
                model.eval()
                for episode in range(episode_count):
                    print('\tEpisode:', episode)
                    random.seed(1)
                    run(model, player_type=player_type, dual=False)
                    germs = []
            # learn from samples
            loss = reinforce(model, optimizer)
            losses.append(loss)
            print('Loss:', loss, '\n')
            optimizer.zero_grad()
            torch.save(model, './model/reinforced.pt')

        save_graph(losses)

    else:  # human or imitation
        score = run(model, player_type=player_type, dual=False)
        print('Score:', score)

    raise SystemExit