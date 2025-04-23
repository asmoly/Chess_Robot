import serial
import threading
import json
import keyboard
from math import pi

class ArmController:
    def __init__(self):
        # Radians
        self.arm_angles = [0, 0, 0, 0, 0, 0]
        
        self.arm_serial = serial.Serial('COM3', baudrate=115200, dsrdtr=None)
        self.arm_serial.setRTS(False)
        self.arm_serial.setDTR(False)

    def read_serial(self):
        while True:
            data = self.arm_serial.readline().decode('utf-8')
            if data:
                try: # Usable data
                    data = json.loads(data)
                    self.arm_angles = [data["b"], data["s"], data["e"], data["t"], data["r"], data["g"]]
                except:
                    pass

    # Gripper: 0 = open, 1 = closed
    # a, b, c, d, e are in degrees
    def set_joint_angles(self, a, b, c, d, e, gripper=0):
        command = '{"T":122,' + '"b":' + str(a) + ',"s":' + str(b) + ',"e":' + str(c) + ',"t":' + str(d) + ',"r":' + str(e) + ',"h":' + str(int(140 + gripper*40)) + ',"spd":' + str(100) + ',"acc":' + str(100) + '}'
        self.arm_serial.write(command.encode() + b'\n')
        self.arm_angles = [a, b, c, d, e, int(140 + gripper*40)]

    def set_torque_off(self):
        command = '{"T":210,"cmd":0}'
        self.arm_serial.write(command.encode() + b'\n')

    def set_torque_on(self):
        command = '{"T":210,"cmd":1}'
        self.arm_serial.write(command.encode() + b'\n')

    def emergency_stop(self):
        command = '{"T":0}'
        self.arm_serial.write(command.encode() + b'\n')

    def reset_emergency_flag(self):
        command = '{"T":999}'
        self.arm_serial.write(command.encode() + b'\n')

    def reset_arm(self):
        self.reset_emergency_flag()
        self.set_joint_angles(0, 0, 90, 0, 0, gripper=1)

    # Gripper: 0 = open, 1 = closed
    def set_gripper_pos(self, gripper):
        command = '{"T":122,' + '"b":' + str(self.arm_angles[0]) + ',"s":' + str(self.arm_angles[1]) + ',"e":' + str(self.arm_angles[2]) + ',"t":' + str(self.arm_angles[3]) + ',"r":' + str(self.arm_angles[4]) + ',"h":' + str(int(140 + gripper*40)) + ',"spd":' + str(30) + ',"acc":' + str(30) + '}'
        self.arm_serial.write(command.encode() + b'\n')

    def start_pos_rc(self, stop_key="s"):
        print("Starting recording")
        self.set_torque_off()
        
        while True:
            data = self.arm_serial.readline().decode('utf-8')
            if data and keyboard.is_pressed(stop_key) == False:
                try: # Usable data
                    data = json.loads(data)
                    self.arm_angles = [int(data["b"]*180/pi), int(data["s"]*180/pi), int(data["e"]*180/pi), int(data["t"]*180/pi), int(data["r"]*180/pi), int(data["g"]*180/pi)]
                    print(self.arm_angles)
                except:
                    print("No data")
                    pass
            elif keyboard.is_pressed(stop_key):
                print("Finished Recording")
                print(self.arm_angles)
                self.set_torque_on()
                return self.arm_angles

arm = ArmController()
arm.set_torque_on()
arm.reset_arm()
#arm.set_joint_angles(23, 48, 97, -1, 0, gripper=1)
#arm.start_pos_rc()