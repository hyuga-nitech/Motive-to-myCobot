import serial
import math
import numpy as np
import threading
import time
from Optitrack import NatNetClient
from pymycobot.mycobot import MyCobot

################################################################################
# ----- Setting Parameter ----- #
myCobotPort  = 'COM8'

RigidBodyID  = 3
RigidBodyNum = 3

MovingLimit = 100   #int[mm]

# ----- Optional Setting parameter ----- #
InitCoordinatelist = [28.0,-68.5,342.2,-84.49,-13.57,-99.57] #list:[x,y,z,rx,ry,rz]

baudrate = 115200    #default:115200
speed = 100          #velocity int:0~100
mode = 0             #mode 0:angluar, 1:linear

# ----- Process parameter ----- #
LoopCount = 0
isMoving  = False

################################################################################

class OptiTrackStreamingManager:
    # ----- Optitrack streaming address ----- #
    serverAddress	= '133.68.35.155'
    localAddress	= '133.68.35.155'

    position = {}	# dict { 'RigidBodyN': [x, y, z] }.  Unit = [m]
    rotation = {}	# dict { 'RigidBodyN': [x, y, z, w]}. 
    
    def __init__(self,defaultRigidBodyNum: int = 1):
        for i in range(defaultRigidBodyNum):
            self.position['RigidBody'+str(i+1)] = np.zeros(3)
            self.rotation['RigidBody'+str(i+1)] = np.zeros(4)

    # This is a callback function that gets connected to the NatNet client and called once per mocap frame.
    def receive_new_frame(self, data_dict):
        order_list=[ "frameNumber", "markerSetCount", "unlabeledMarkersCount", "rigidBodyCount", "skeletonCount",
                    "labeledMarkerCount", "timecode", "timecodeSub", "timestamp", "isRecording", "trackedModelsChanged" ]
        dump_args = False
        if dump_args == True:
            out_string = "    "
            for key in data_dict:
                out_string += key + "="
                if key in data_dict :
                    out_string += data_dict[key] + " "
                out_string+="/"
            print(out_string)

    # This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame
    def receive_rigid_body_frame( self, new_id, position, rotation ):
        if 'RigidBody'+str(new_id) in self.position:
            self.position['RigidBody'+str(new_id)] = np.array(position)
            self.rotation['RigidBody'+str(new_id)] = np.array(rotation)

    def stream_run(self):
        streamingClient = NatNetClient.NatNetClient(serverIP=self.serverAddress, localIP=self.localAddress)
        
        streamingClient.new_frame_listener = self.receive_new_frame
        streamingClient.rigid_body_listener = self.receive_rigid_body_frame
        streamingClient.run()

class MotionManager:
    def __init__(self,defaultRigidBodyNum: int) -> None:
        self.optiTrackStreamingManager = OptiTrackStreamingManager(defaultRigidBodyNum = RigidBodyNum)
        streamingThread = threading.Thread(target=self.optiTrackStreamingManager.stream_run)
        streamingThread.setDaemon(True)
        streamingThread.start()

    def LocalPosition(self, loopCount: int = 0):
        dictPos = {}
        dictPos = self.optiTrackStreamingManager.position
        Pos     = dictPos['RigidBody'+str(RigidBodyID)]
        return Pos

    def LocalRotation(self, loopCount: int = 0):
        dictRot = {}
        dictRot = self.optiTrackStreamingManager.rotation
        Rot     = dictRot['RigidBody'+str(RigidBodyID)]
        return Rot

class MotionCalculator:
    def __init__(self,defaultRigidBodyNum: int) -> None:
        self.originalPositions = np.zeros(3)
        self.inversedMatrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

        self.Position = np.zeros(3)
        self.Rotations = np.array([0,0,0,1])

    def GetTransform(self,position,rotation):
        relativePos = self.GetRelativePosition(position)
        relativeRot = self.GetRelativeRotation(rotation)

        return relativePos , relativeRot

    def SetOriginalPosition(self, position) -> None:
        self.originalPositions = position

    def SetInversedMatrix(self, rotation) -> None:
        q = rotation
        qw, qx, qy, qz = q[3], q[1], q[2], q[0]
        mat4x4 = np.array([ [qw, -qy, qx, qz],
                            [qy, qw, -qz, qx],
                            [-qx, qz, qw, qy],
                            [-qz,-qx, -qy, qw]])
        self.inversedMatrix = np.linalg.pinv(mat4x4)

    def GetRelativePosition(self, position):
        relativePos = position - self.originalPositions

        return relativePos

    def GetRelativeRotation(self, rotation):
        Q_relativeRot = np.dot(self.inversedMatrix, rotation)
        E_relativeRot = self.Quaternion2Euler(np.array(Q_relativeRot))

        return E_relativeRot

    def Quaternion2Euler(self, q, isDeg: bool = True):
        """
        Calculate the Euler angle from the Quaternion.
        
        Rotation matrix
        |m00 m01 m02 0|
        |m10 m11 m12 0|
        |m20 m21 m22 0|
        | 0   0   0  1|

        Parameters
        ----------
        q: np.ndarray
            Quaternion.
            [x, y, z, w]
        isDeg: (Optional) bool
            Returned angles are in degrees if this flag is True, else they are in radians.
            The default is True.
        
        Returns
        ----------
        rotEuler: np.ndarray
            Euler angle.
            [x, y, z]
        """
        
        qx = q[0]
        qy = q[1]
        qz = q[2]
        qw = q[3]

        # 1 - 2y^2 - 2z^2
        m00 = 1 - (2 * qy**2) - (2 * qz**2)
        # 2xy + 2wz
        m01 = (2 * qx * qy) + (2 * qw * qz)
        # 2xz - 2wy
        m02 = (2 * qx * qz) - (2 * qw * qy)
        # 2xy - 2wz
        m10 = (2 * qx * qy) - (2 * qw * qz)
        # 1 - 2x^2 - 2z^2
        m11 = 1 - (2 * qx**2) - (2 * qz**2)
        # 2yz + 2wx
        m12 = (2 * qy * qz) + (2 * qw * qx)
        # 2xz + 2wy
        m20 = (2 * qx * qz) + (2 * qw * qy)
        # 2yz - 2wx
        m21 = (2 * qy * qz) - (2 * qw * qx)
        # 1 - 2x^2 - 2y^2
        m22 = 1 - (2 * qx**2) - (2 * qy**2)

        # 回転軸の順番がX->Y->Zの固定角(Rz*Ry*Rx)
        # if m01 == -1:
        # 	tx = 0
        # 	ty = math.pi/2
        # 	tz = math.atan2(m20, m10)
        # elif m20 == 1:
        # 	tx = 0
        # 	ty = -math.pi/2
        # 	tz = math.atan2(m20, m10)
        # else:
        # 	tx = -math.atan2(m02, m00)
        # 	ty = -math.asin(-m01)
        # 	tz = -math.atan2(m21, m11)

        # 回転軸の順番がX->Y->Zのオイラー角(Rx*Ry*Rz)
        if m02 == 1:
            tx = math.atan2(m10, m11)
            ty = math.pi/2
            tz = 0
        elif m02 == -1:
            tx = math.atan2(m21, m20)
            ty = -math.pi/2
            tz = 0
        else:
            tx = -math.atan2(-m12, m22)
            ty = -math.asin(m02)
            tz = -math.atan2(-m01, m00)

        if isDeg:
            tx = np.rad2deg(tx)
            ty = np.rad2deg(ty)
            tz = np.rad2deg(tz)

        rotEuler = np.array([tx, ty, tz])
        return rotEuler

if __name__ == '__main__':
    mycobot = MyCobot(myCobotPort,baudrate)
    mycobot.send_coords(InitCoordinatelist,speed,mode)

    time.sleep(3)

    motionManager = MotionManager(RigidBodyNum)
    motionCalculator = MotionCalculator(RigidBodyNum)

    try:
        while True:
            if isMoving:
                position = motionManager.LocalPosition(loopCount = LoopCount)
                rotation = motionManager.LocalRotation(loopCount = LoopCount)

                mycobotPos,mycobotRot = MotionCalculator.GetTransform(position, rotation)
                
                mycobotPos = mycobotPos * 1000

                mycobotTransform = [mycobotPos[2] + InitCoordinatelist[0], 
                                    mycobotPos[0] + InitCoordinatelist[1],
                                    mycobotPos[1] + InitCoordinatelist[2], 
                                    mycobotRot[2] + InitCoordinatelist[3], 
                                    -1 * mycobotRot[0] + InitCoordinatelist[4], 
                                    -1 * mycobotRot[1] + InitCoordinatelist[5]]
                
                diffX = mycobotTransform[0] - beforeTransform[0]
                diffY = mycobotTransform[1] - beforeTransform[1]
                diffZ = mycobotTransform[2] - beforeTransform[2]

                beforeTransform = mycobotTransform

                if abs(diffX) > MovingLimit or abs(diffY) > MovingLimit or abs(diffZ) > MovingLimit:
                    isMoving = False
                    print('[ERROR] >> A rapid movement has occurred.')
                
                else:
                    mycobot.send_coords(mycobotTransform,speed,mode)

                LoopCount += 1

            else:
                position = motionManager.LocalPosition(loopCount = LoopCount)
                rotation = motionManager.LocalRotation(loopCount = LoopCount)

                motionCalculator.SetOriginalPosition(position)
                motionCalculator.SetInversedMatrix(rotation)

                mycobotPos, mycobotRot = motionCalculator.GetTransform(position, rotation)
                
                mycobotPos = mycobotPos * 1000

                mycobotTransform = [mycobotPos[2] + InitCoordinatelist[0], 
                                    mycobotPos[0] + InitCoordinatelist[1],
                                    mycobotPos[1] + InitCoordinatelist[2], 
                                    mycobotRot[2] + InitCoordinatelist[3], 
                                    -1 * mycobotRot[0] + InitCoordinatelist[4], 
                                    -1 * mycobotRot[1] + InitCoordinatelist[5]]
                
                beforeTransform  = mycobotTransform

                isMoving = True

    except:
        import traceback
        traceback.print_exc()
