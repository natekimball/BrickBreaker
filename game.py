import pygame
import sys
import random
import numpy as np
from DQNagent import Agent
import matplotlib.pyplot as plt


WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (30, 30, 230)
RED = (230, 30, 30)
GREEN = (30, 230, 30)
YELLOW = (230, 230, 30)
PURPLE = (230, 30, 230)
COLORS = [BLUE, RED, GREEN, YELLOW, PURPLE]

PADDLE_WIDTH, PADDLE_HEIGHT = 80, 15
BALL_WIDTH, BALL_HEIGHT = 15, 15
BRICK_WIDTH, BRICK_HEIGHT = 75, 20

INTERACTIVE = '-i' in sys.argv or '--interactive' in sys.argv
SPEEDUP = 1 if INTERACTIVE else 3

def launch_game(agent, id=None):
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    state = initialize()
    paddle, bricks, ball, ball_dx, ball_dy = state
    score = 0
    i = 0
    last_collision = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if INTERACTIVE:
            keys = pygame.key.get_pressed()
            left, right = keys[pygame.K_LEFT], keys[pygame.K_RIGHT]
        elif i % SPEEDUP == 0:
            state_array = agent.shape(state)
            Q_vals, actions = agent.get_action(state_array)
            # actions = agent.get_action(state_array)
            left, right = actions
        else:
            left, right = False, False
            

        if left and paddle.left > 0:
            paddle.left -= 5 * SPEEDUP
        if right and paddle.right < WIDTH:
            paddle.right += 5 * SPEEDUP

        ball.left += ball_dx
        ball.top += ball_dy

        reward = 0
        ball_center = ball.left + ball.width // 2
        if ball.left < 0 or ball.right > WIDTH:
            ball_dx *= -1
        if ball.top < 0:
            ball_dy *= -1
        collision = 0
        if ball.colliderect(paddle) and ball_dy > 0:
            last_collision = i
            collision = 1
            paddle_third = paddle.left + paddle.width // 3
            paddle_2nd_third = paddle.left + 2*paddle.width // 3
            d = np.array([ball_dx, ball_dy])
            # formula for vector reflection: r = d - (2*d.n) * n
            if ball_center < paddle_third:
                n = np.array([-.196, -.981])
            elif ball_center > paddle_2nd_third:
                n = np.array([.196, -.981])
            else:
                n = np.array([0, -1])
            ball_dx, ball_dy = d - (2*np.dot(d, n) * n)
            ball_dx, ball_dy = ball_dx*1.01, ball_dy*1.01
        else:
            for brick,c in bricks:
                if ball.colliderect(brick):
                    last_collision = i
                    if ball_center < brick.left or ball_center > brick.right:
                        ball_dx *= -1
                    else:
                        ball_dy *= -1
                    reward += 1
                    bricks.remove((brick,c))
                    break

        if np.isclose(ball_dy, 0):
            ball_dy = .1
        
        score += reward
        reward += collision

        if ball.bottom > HEIGHT or i - last_collision > 600 or np.isclose(ball_dy, 0):
            pygame.quit()
            if not INTERACTIVE:
                agent.lost(state_array, actions, Q_vals)
            return score
        
        if not bricks:
            _, bricks, ball, ball_dx, ball_dy = initialize()

        if not INTERACTIVE and i % SPEEDUP == 0:
            agent.update(state_array, actions, reward, agent.shape(state), Q_vals)
        
        redraw_window(win,state, score, id, agent.epsilon if agent is not None else None)
        i+=1

def initialize():
    paddle = pygame.Rect(WIDTH // 2, HEIGHT - PADDLE_HEIGHT - 10, PADDLE_WIDTH, PADDLE_HEIGHT)
    ball = pygame.Rect(random.randint(0,WIDTH - BALL_WIDTH), HEIGHT // 2, BALL_WIDTH, BALL_HEIGHT)
    # ball_dx, ball_dy = 3 * SPEEDUP, -3 * SPEEDUP
    ball_dx = 3 if random.randint(0,1) else -3
    ball_dy = -3
    bricks = [(pygame.Rect(j * (BRICK_WIDTH + 5), i * (BRICK_HEIGHT + 5), BRICK_WIDTH, BRICK_HEIGHT), COLORS[i % len(COLORS)])
           for i in range(5) for j in range(WIDTH // (BRICK_WIDTH + 5))]
    return paddle, bricks, ball, ball_dx, ball_dy
    
def redraw_window(win, state, score, game_num=None, epsilon=None):
    paddle, bricks, ball, _, _ = state
    win.fill(BLACK)
    pygame.draw.rect(win, BLUE, paddle)
    pygame.draw.rect(win, BLUE, ball)
    for brick, color in bricks:
        pygame.draw.rect(win, color, brick)

    font = pygame.font.Font(None, 36)
    text = font.render("Score: " + str(score), True, WHITE)
    win.blit(text, (10,10))
    
    if game_num is not None:
        font = pygame.font.Font(None, 24)
        text = font.render("Game: " + str(game_num), True, WHITE)
        win.blit(text, (WIDTH-100,HEIGHT-30))
    
    if epsilon is not None:
        font = pygame.font.Font(None, 24)
        text = font.render(f"ε = {epsilon:.3f}", True, WHITE)
        win.blit(text, (10,HEIGHT-30))

    pygame.display.flip()
    pygame.time.delay(10)

def get_arg(key, default):
    if type(key) == list:
        for k in key:
            if k in sys.argv:
                return sys.argv[sys.argv.index(k) + 1]
    elif key in sys.argv:
        return sys.argv[sys.argv.index(key) + 1]
    return default

if __name__ == "__main__":
    if INTERACTIVE:
        launch_game(None)
    else:
        num_games = int(get_arg(['-n','--num-games'], 100))
        gamma = float(get_arg(['-g','--gamma'], .9))
        epsilon_decay = float(get_arg(['-d','--epsilon-decay'], .999)) # max(.999,1-1/(10*num_games))))
        learning_rate=float(get_arg(['-l','--learning-rate'], .001))
        replay = '--no-replay' not in sys.argv
        batch_size = int(get_arg(['-b','--batch-size'], 32))
        memory_size = int(get_arg(['-m','--memory'], 1024))
        model_dir = get_arg(['-f','--model-file'], None)
        save_dir = get_arg(['-s','--save-dir'], "brickbreaker_model.pt")
        
        agent = Agent(gamma, epsilon_decay,learning_rate, model_dir=model_dir, replay=replay, batch_size=batch_size, memory_size=memory_size)
        scores = [0]*num_games
        moving_avgs = [0]*num_games
        for i in range(num_games):
            # Game loop 
            score = launch_game(agent, i)
            print(f"game {i} - {score}")
            scores[i] = score
            moving_avgs[i] = (moving_avgs[i-1]*min(i,10) + score - (scores[i-10] if i > 10 else 0))/min(i+1,10)

            # plt.cla()
            # plt.plot(scores[:i+1], label='Scores', color='blue')
            # plt.plot(moving_avgs[:i+1], label='Moving average', color='red')
            # plt.xlabel('Game #')
            # plt.ylabel('Score')
            # plt.xlim([0, num_games])  # Set the limit of x-axis here
            # plt.legend()
            # plt.pause(0.01)
        
        games = np.arange(1,num_games+1)
        coeff = np.polyfit(np.log(games), scores, 1)
        lin_func = np.poly1d(coeff)
        y_pred = lin_func(np.log(games))

        plt.plot(games, scores, label=f'Scores')
        plt.plot(games, moving_avgs, label='Moving average')
        plt.plot(games, y_pred, label=f'Score Regression, y = {coeff[0]:.2f}*ln(x) + {coeff[1]:.2f} (R^2 = {np.corrcoef(scores, y_pred)[0,1]**2:.3f})')
        plt.title('DQN performance over time')
        plt.xlabel('Game number')
        plt.ylabel('Score')
        plt.legend()
        # plt.show()
        plt.savefig('score_plot.png')

        agent.save(save_dir)