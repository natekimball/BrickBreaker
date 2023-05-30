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

interactive = '-i' in sys.argv or '--interactive' in sys.argv

def launch_game(agent):
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    state = initialize()
    paddle, bricks, ball, ball_dx, ball_dy = state
    while True:
        score = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if interactive:
            keys = pygame.key.get_pressed()
            left, right = keys[pygame.K_LEFT], keys[pygame.K_RIGHT]
        else:
            state_array = agent.shape(state)
            Q_vals, actions = agent.get_action(state_array)
            print(actions)
            left, right = actions
    
        if left and paddle.left > 0:
            paddle.left -= 5
        if right and paddle.right < WIDTH:
            paddle.right += 5

        ball.left += ball_dx
        ball.top += ball_dy

        reward = 0
        ball_center = ball.left + ball.width // 2
        if ball.left < 0 or ball.right > WIDTH:
            ball_dx *= -1
        if ball.top < 0:
            ball_dy *= -1
        if ball.colliderect(paddle) and ball_dy > 0:
            reward += .5
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
                    if ball_center < brick.left or ball_center > brick.right:
                        ball_dx *= -1
                    else:
                        ball_dy *= -1
                    reward += 1
                    bricks.remove((brick,c))
                    break

        # if np.isclose(ball_dy, 0):
            # ball_dy = .1
        
        score += reward

        if ball.bottom > HEIGHT or np.isclose(ball_dy, 0):
            pygame.quit()
            if not interactive:
                agent.lost(Q_vals, state_array, actions)
            return score
        
        if not bricks:
            _, bricks, ball, ball_dx, ball_dy = initialize()

        if not interactive:
            agent.update(Q_vals, state_array, actions, reward, agent.shape(state))
        
        redraw_window(win,state)

def initialize():
    paddle = pygame.Rect(WIDTH // 2, HEIGHT - PADDLE_HEIGHT - 10, PADDLE_WIDTH, PADDLE_HEIGHT)
    ball = pygame.Rect(random.randint(0,WIDTH - BALL_WIDTH), HEIGHT // 2, BALL_WIDTH, BALL_HEIGHT)
    ball_dx, ball_dy = 3, -3
    bricks = [(pygame.Rect(j * (BRICK_WIDTH + 5), i * (BRICK_HEIGHT + 5), BRICK_WIDTH, BRICK_HEIGHT), COLORS[i % len(COLORS)])
           for i in range(5) for j in range(WIDTH // (BRICK_WIDTH + 5))]
    return paddle, bricks, ball, ball_dx, ball_dy
    
def redraw_window(win, state):
    paddle, bricks, ball, _, _ = state
    win.fill(BLACK)
    pygame.draw.rect(win, BLUE, paddle)
    pygame.draw.rect(win, BLUE, ball)
    for brick, color in bricks:
        pygame.draw.rect(win, color, brick)

    pygame.display.flip()
    pygame.time.delay(10)
    
if __name__ == "__main__":
    if interactive:
        launch_game(None)
    else:
        num_games = 1000
        agent = Agent()
        scores = [0]*num_games
        moving_avgs = [0]*num_games
        for i in range(num_games):
            # Game loop
            score = launch_game(agent)
            print(f"game {i} - {score}")
            scores[i] = score
            moving_avgs[i] = (moving_avgs[i-1]*min(i,10) + score - (scores[i-10] if i > 10 else 0))/min(i+1,10)

        games = np.arange(num_games)

        scores_exp = np.exp(scores)
        print(scores)
        print(scores_exp)
        coeff = np.polyfit(games, scores_exp, 1)
        lin_func = np.poly1d(coeff)
        y_pred = lin_func(games)

        plt.plot(scores, label='Score')
        plt.plot(moving_avgs, label='Moving average')
        plt.plot(np.log(y_pred), label='Score Regression')
        # plt.plot(np.exp(np.log(agent.epsilon_decay)*np.arange(num_games)), label='Epsilon')
        plt.title('DQN performance over time')
        plt.xlabel('Game number')
        plt.ylabel('Score')
        plt.legend()
        plt.show()
        plt.savefig('score_plot.png')

        agent.save("brickbreaker_model")