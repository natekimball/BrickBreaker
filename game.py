import pygame
import sys
import random
import numpy as np
# from DQNagent import Agent


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

        # keys = pygame.key.get_pressed()
        # if keys[pygame.K_LEFT] and paddle.left > 0:
        #     paddle.left -= 5
        # if keys[pygame.K_RIGHT] and paddle.right < WIDTH:
        #     paddle.right += 5
        
        actions = agent.get_action(state)
        if actions[0] == 1 and paddle.left > 0:
            paddle.left -= 5
        if actions[1] == 1 and paddle.right < WIDTH:
            paddle.right += 5

        # Ball movement
        ball.left += ball_dx
        ball.top += ball_dy

        reward = 0
        # Collisions
        ball_center = ball.left + ball.width // 2
        if ball.left < 0 or ball.right > WIDTH:
            ball_dx *= -1
        if ball.top < 0:
            ball_dy *= -1
        if ball.colliderect(paddle) and ball_dy > 0:
            paddle_third = paddle.left + paddle.width // 3
            paddle_2nd_third = paddle.left + 2*paddle.width // 3
            d = np.array([ball_dx, ball_dy])
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
        if np.isclose(ball_dy, 0):
            pygame.quit()
            # ball_dy = .1

        score += reward
        if not bricks:
            _, bricks, ball, ball_dx, ball_dy = initialize()
        
        # Game over
        if ball.bottom > HEIGHT:
            pygame.quit()
            agent.loss(state, actions)
            return score
            # sys.exit()
        
        agent.update(state, actions, reward)
        
        # Draw everything
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
    num_games = 1000000
    agent = Agent()
    scores = []
    for i in range(num_games):
        # Game loop
        scores.append(launch_game(agent))