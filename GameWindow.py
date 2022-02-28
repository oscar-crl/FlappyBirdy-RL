import math
from tkinter import Canvas

import numpy as np

from GameInfo import GameInfo
from Pipe import Pipe
from Player import Player

from Score import Score

TICK_DURATION = math.floor(1 / 60)
NEW_PIPE = 2000


class GameWindow(Canvas):

    def init_callbacks(self):
        self.player.jump()

    def update_player_pos(self):
        self.delete(self.crappy)
        self.crappy = self.player.draw(self)

    def update_pipes_pos(self):
        for pipe in self.pipes:
            pipe.tick()
            if pipe.is_out_of_bounds():
                self.pipes.remove(pipe)
            else:
                pipe.draw(self)

    def check_lose(self):
        results = self.find_overlapping(*self.player.get_hitbox())
        # self.find_overlapping(*self.pipes[-1].get_hitbox(1))
        self.delete(self.hitbox_check)
        # self.delete(self.hitbox_pipes)
        # self.hitbox_check = self.create_rectangle(*self.player.get_hitbox(), fill="red", tag="hitbox")
        # self.hitbox_pipes = self.create_rectangle(*self.pipes[-1].get_hitbox(1), fill="blue", tag="hitboxpipe")

        if self.player.is_out_of_bounds():
            self.stop = True

        for result in results:
            if "pipe" in self.gettags(result):
                self.stop = True

    def update_pos(self):
        self.update_player_pos()
        self.update_pipes_pos()
        self.score.draw(self)

    def validate_pipe(self, pipe):
        if self.player.pos.x > pipe.positions[0].x:
            pipe.score_validated = True
            self.score.score += 1

    def check_score(self):
        for pipe in self.pipes:
            if pipe.score_validated:
                continue
            self.validate_pipe(pipe)

    def check_new_pipe(self):
        last_pipe = self.pipes[-1]

        if self.get_distance_from_pipe() <= 0:
            self.add_new_pipe()
        #if last_pipe.positions[0].x <= GameInfo.scree_size.x / 2:
        #   self.add_new_pipe()

    def get_distance_from_hole_pos(self):
        # print('pos {} - size {}'.format(self.pipes[-1].positions[0].y, self.pipes[-1].size.y))
        return int(self.pipes[-1].positions[0].y - self.pipes[-1].size.y / 2) - self.player.pos.y

    def get_distance_from_pipe(self):
        last_pipe = self.pipes[-1]
        pos = (last_pipe.positions[0].x + (last_pipe.size.x / 2) + (self.player.size.x / 2)) - self.player.pos.x

        return pos if pos >= 0 else 0

    def get_state(self):
        h_dist = self.get_distance_from_pipe()
        next_pipe_pos = self.pipes[-1].positions[1].y + self.pipes[-1].size.y / 2
        player_pos = self.player.get_player_pos()
        wanted_pos = next_pipe_pos + (GameInfo.window_size.y / 6)
        v_dist = int(wanted_pos - player_pos)

        # print(h_dist / (GameInfo.window_size.x / 2), v_dist / GameInfo.window_size.y)

        return np.array([h_dist, v_dist])

    def draw_visualizer(self):
        state = self.get_state()
        next_pipe_pos = self.pipes[-1].positions[1].y + self.pipes[-1].size.y / 2
        player_pos = self.player.get_player_pos()
        wanted_pos = next_pipe_pos + (GameInfo.window_size.y / 4) - self.player.size.y / 2

        for draw_data in self.visualizer:
            self.delete(draw_data)
        self.visualizer.append(self.create_line(self.player.pos.x, self.player.pos.y, self.player.pos.x + state[0], self.player.pos.y, fill="blue"))
        self.visualizer.append(self.create_line(self.player.pos.x, self.player.pos.y, self.pipes[-1].positions[0].x, self.player.pos.y + state[1], fill="red"))

    def tick(self, action):
        # if self.stop == False:
        #    self.after(TICK_DURATION, self.tick)

        self.update_pos()
        self.player.tick(action)
        self.check_lose()
        self.check_score()
        self.check_new_pipe()
        # self.draw_visualizer()
        return self.get_state()
        # print('Y pos {} - Distance X {} - Y hole {}'.format(self.player.pos.y, self.get_distance_from_pipe(), self.get_pipe_hole_position()))

    def add_new_pipe(self):
        # self.after(NEW_PIPE, self.add_new_pipe)
        if self.player.started == False:
            return
        self.pipes.append(Pipe())

    def reset(self):

        for pipe in self.pipes:
            for tag in pipe.tags:
                self.delete(tag)
        self.score.score = 0
        self.pipes = [Pipe()]
        self.stop = False
        self.player = Player()

    def __init__(self, root):
        super().__init__(width=480, height=640, background="black", highlightthickness=0)
        self.screen_size = (480, 640)
        self.root = root
        self.player = Player()
        self.pack()
        self.pipes = [Pipe()]
        self.pipe_img = None
        self.flipped_pipe_img = None
        self.crappy = -1
        self.stop = False
        self.hitbox_check = None
        self.hitbox_pipes = None
        self.score = Score()
        self.visualizer = []
        self.i = 0
