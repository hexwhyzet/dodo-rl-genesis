#!/usr/bin/env python3
"""Print all gamepad axis/button values to figure out the mapping."""
import pygame

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No joystick found.")
    raise SystemExit

joy = pygame.joystick.Joystick(0)
joy.init()
print(f"Controller: {joy.get_name()}")
print(f"Axes: {joy.get_numaxes()}  Buttons: {joy.get_numbuttons()}")
print("Move sticks and press buttons. Ctrl+C to quit.\n")

pygame.display.set_mode((1, 1), pygame.NOFRAME)

prev = {}
try:
    while True:
        pygame.event.pump()
        state = {}
        for i in range(joy.get_numaxes()):
            v = joy.get_axis(i)
            state[f"axis{i}"] = f"{v:+.2f}"
        for i in range(joy.get_numbuttons()):
            if joy.get_button(i):
                state[f"btn{i}"] = "ON"
        if state != prev:
            print("  ".join(f"{k}={v}" for k, v in state.items()) or "(idle)")
            prev = state
        pygame.time.wait(50)
except KeyboardInterrupt:
    pass
pygame.quit()
