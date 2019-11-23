import pygame
import time
import numpy as np
import math
import sys

import collisions
import event
import gamestate
import graphics
import config
import ball

sys.path.append('../agent')
import agent


# print(type(game))
# print(type(game.balls))
# print(type([ball for ball in game.balls][0]))

# <class 'gamestate.GameState'>
# <class 'pygame.sprite.Group'>
# <class 'ball.BallSprite'>

# Some Parameters to control the game
# set player_2 to None for single player game
player_1 = gamestate.PlayerType.Bot
player_2 = None # gamestate.PlayerType.Human
# More Player Types can be added for Bot Algorithms

was_closed = False # True when the game is closed from menu
while not was_closed:

    ag = agent.Agent()
    # start a new game and take input from menu
    game = gamestate.GameState()
    # button_pressed = graphics.draw_main_menu(game)
    
    # Play game Button
    button_pressed = config.play_game_button

    # if the play game button is pressed
    if button_pressed == config.play_game_button:
        print("In!")
        game.start_pool()
        ag.setInitState(game.getGameState())
        if(config.game_count == config.max_game_count):
            ag.saveModel()
            config.game_count = 0
        events = event.events()

        # keep playing the game till it's over or is exit
        while not (events["closed"] or game.is_game_over or events["quit_to_main_menu"]):
            # comes here at every render (not after every turn)
            # print("Waiting...")
            events = event.events()
            collisions.resolve_all_collisions(game.balls, game.holes, game.table_sides)
            game.redraw_all()

            # if the balls are done moving, play next turn
            if game.all_not_moving():
                game.check_pool_rules()
                if config.graphics_flag:
                    game.cue.make_visible(game.current_player) # to hide cue all time

                # player_1 always plays for a single player game
                if player_2 is None:
                    game.current_player = gamestate.Player.Player1

                while not (
                    (events["closed"] or events["quit_to_main_menu"]) or game.is_game_over) and game.all_not_moving():
                    game.redraw_all()
                    events = event.events()
                    # determine current player's type
                    curr_player_type = None
                    if game.current_player == gamestate.Player.Player1:
                        curr_player_type = player_1
                    else:
                        curr_player_type = player_2

                    # if current player is a Bot
                    if curr_player_type == gamestate.PlayerType.Bot:
                        if game.all_not_moving():
                            ag.getState(game.getGameState())
                            a,d = ag.getAction()
                            game.cue.botPlay(a,d)
                        elif game.can_move_white_ball and game.white_ball.is_clicked(events):
                            game.white_ball.is_active(game, game.is_behind_line_break())

                    else: # if current player is a Human
                        if game.cue.is_clicked(events):
                            game.cue.cue_is_active(game, events)
                        elif game.can_move_white_ball and game.white_ball.is_clicked(events):
                            game.white_ball.is_active(game, game.is_behind_line_break())
        was_closed = events["closed"]

    # if the close game button is pressed
    if button_pressed == config.exit_button:
        was_closed = True

pygame.quit()
