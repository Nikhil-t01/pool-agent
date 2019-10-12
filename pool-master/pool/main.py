import pygame
import time
import numpy as np
import math

import collisions
import event
import gamestate
import graphics
import config

# print(type(game))
# print(type(game.balls))
# print(type([ball for ball in game.balls][0]))

# <class 'gamestate.GameState'>
# <class 'pygame.sprite.Group'>
# <class 'ball.BallSprite'>

was_closed = False
while not was_closed:
    game = gamestate.GameState()
    button_pressed = graphics.draw_main_menu(game)

    if button_pressed == config.play_game_button:
        game.start_pool()
        events = event.events()
        while not (events["closed"] or game.is_game_over or events["quit_to_main_menu"]):
            events = event.events()
            collisions.resolve_all_collisions(game.balls, game.holes, game.table_sides)
            game.redraw_all()

            if game.all_not_moving():
                game.check_pool_rules()
                game.cue.make_visible(game.current_player)

                while not (
                    (events["closed"] or events["quit_to_main_menu"]) or game.is_game_over) and game.all_not_moving():
                    game.redraw_all()
                    events = event.events()
                    if game.current_player == gamestate.Player.Player1:
                        # BOT
                        if game.all_not_moving():
                            print(len(game.getGameState()))
                            # print(game.getGameState())
                            angle = np.random.uniform(-math.pi,math.pi)
                            distance = np.random.randint(30,100)
                            game.cue.botPlay(angle,distance)
                        elif game.can_move_white_ball and game.white_ball.is_clicked(events):
                            game.white_ball.is_active(game, game.is_behind_line_break())
                    else:
                        # HUMAN
                        if game.cue.is_clicked(events):
                            if game.all_not_moving():
                                print(len(game.getGameState()))
                            game.cue.cue_is_active(game, events)
                        elif game.can_move_white_ball and game.white_ball.is_clicked(events):
                            game.white_ball.is_active(game, game.is_behind_line_break())
        was_closed = events["closed"]

    if button_pressed == config.exit_button:
        was_closed = True

pygame.quit()
