
import sys
import os

import numpy as np
from functions import LOGGER


class HDAS(object):
    def __init__(self, PathFileName: str, load: bool = True):
        """
        :param PathFileName: Ruta absoluta del archivo
        :param load: booleano para indicar si quieres que se carge o no
        :param RawData: Hay que indicar si es RawData o no
        """
        super().__init__()
        self.PathFileName = PathFileName
        self.N_processed_Points, self.N_Time_Samples = None, None
        self.fHeaderSize = None
        self.Headers = None
        self.FileHeader25 = None
        self.Spatial_Sampling_Meters = None
        self.Fiber_Length_Monitored_Meters = None
        self.Mode = None
        self.TimeStart = 0
        self.TimeStop = None
        self.FiberStart = 0
        self.FiberStop = None
        self.Data = None
        [self.Fiber_Position_Offset, self.Trigger_Frequency,
            self.FiberRefStart, self.FiberRefStop,
            self.MultiPointPosMeter, self.RefUpdate_Window] = [None, None, None, None, None, None]
        self.average = None
        self.time_partition = None
        self.fiber_partition = None
        self.TimeWindow_seconds = 1
        self.TimeWindow = None
        self.N_TimeWindows = None
        self.is_loaded = False
        self.Sqrt_Ref_Fibre_Length = None
        self.segtotal = None
        self.AdaptedData = None
        self.RawMode = None
        self.ScaleFactor = 2.44*10
        if load:
            self.load()

    def __len__(self):
        """
        :return: Devuelve el numero de puntos de la traza
        """
        return self.Data.shape[0]

    def __copy__(self):
        return HDAS(self.PathFileName)

    def __str__(self):
        return 'HDAS'

    def copy(self):
        return self.__copy__()

    def setPathFileName(self, value):
        self.PathFileName = value

    def set_vars_header(self):
        """
            Get Measurement Settings [Choose strain or temperature settings only (depending on
                whether it is a temperature or strain file)]
        """
        self.fHeaderSize = self.Headers[0]
        self.Spatial_Sampling_Meters = self.Headers[1]
        self.Fiber_Length_Monitored_Meters = self.Headers[3]  # Fiber lenght monitored (not necessarily processed)
        self.Mode = int(self.Headers[101])
        self.average = self.Headers[15]
        self.FileHeader25 = self.Headers[24]
        [self.Headers, self.Fiber_Position_Offset, self.Trigger_Frequency,
         self.FiberRefStart, self.FiberRefStop,
         self.MultiPointPosMeter, self.RefUpdate_Window] = load_header(self.fHeaderSize, self.Headers,
                                                                       self.Spatial_Sampling_Meters)

    def set_time_n_fiber(self):
        self.TimeStart = 0
        self.TimeStop = len(self.Data[1])

    def set_values_time_n_fiber(self, fiber, time):
        self.TimeStart = 0
        self.TimeStop = time
        self.FiberStart = 0
        self.FiberStop = fiber

    def set_time_window(self):
        """
            TimeWindow := Time (seconds) over which the FFT is processed to calculate energy
            M_TimeWindows := Number of time windows over which Acoustic Energy will be calculated for each point
        """
        self.TimeWindow_seconds = self.FileHeader25 / 1000
        self.TimeWindow = int(self.TimeWindow_seconds * self.Trigger_Frequency)
        self.N_TimeWindows = int(np.floor((self.TimeStop-self.TimeStart + 2) / self.TimeWindow))

    def load(self):
        self.loadMatrix()
        self.set_time_n_fiber()
        self.set_time_window()

    def loadMatrix(self):
        """ load data set """
        try:
            [self.Data, self.Headers, self.N_processed_Points, self.N_Time_Samples, self.RawMode] = LoadData(self.PathFileName)

            assert self.Data is not None, "Error loading file .bin"
            if self.Data is None:
                self.is_loaded = False
            else:
                self.is_loaded = True
                self.set_vars_header()

            LOGGER.debug('File {} loaded! '.format(os.path.basename(self.PathFileName)))
        except Exception as e:
            LOGGER.error('Error loading the file. Message error {}'.format(e))


class fileTypeOptions(object):
    def __init__(self, Headers, Spatial_Sampling_Meters):
        super().__init__()
        self.Headers = Headers
        self.Spatial_Sampling_Meters = Spatial_Sampling_Meters
        self.Fiber_Position_Offset = 0
        self.Trigger_Frequency = 0
        self.FiberRefStart = 0
        self.FiberRefStop = 0
        self.MultiPpointPosMeter = 0
        self.RefUpdate_Window = 0

    def strain_raw_data(self):
        self.Fiber_Position_Offset = self.Headers[11]  # Fiber position where strain processing starts
        self.Trigger_Frequency = self.Headers[6] / self.Headers[15] / self.Headers[98]
        self.FiberRefStart = self.Headers[17] + 1
        self.FiberRefStop = self.Headers[19] + 1
        self.MultiPpointPosMeter = self.Headers[41:51] * self.Spatial_Sampling_Meters + self.Fiber_Position_Offset
        self.RefUpdate_Window = self.Headers[23] * self.Headers[6] / self.Headers[15] / 1000
    def temp_raw_data(self):
        self.Fiber_Position_Offset = self.Headers[11]  # Fiber position where strain processing starts
        self.Trigger_Frequency = self.Headers[6] / self.Headers[15] / self.Headers[98]
        self.FiberRefStart = self.Headers[17] + 1
        self.FiberRefStop = self.Headers[19] + 1
        self.MultiPpointPosMeter = self.Headers[51:61] * self.Spatial_Sampling_Meters + self.Fiber_Position_Offset
        self.RefUpdate_Window = self.Headers[40] * self.Headers[6] / self.Headers[32] / 1000
    def strain_map(self):
        self.Trigger_Frequency = self.Headers[6] / self.Headers[15] / self.Headers[98]
        self.FiberRefStart = self.Headers[17] + 1
        self.FiberRefStop = self.Headers[19] + 1
        self.Fiber_Position_Offset = self.Headers[11]
        self.MultiPpointPosMeter = self.Headers[41:51] * self.Spatial_Sampling_Meters + self.Fiber_Position_Offset

    def temp_map(self):
        self.Trigger_Frequency = self.Headers[6] / self.Headers[32] / self.Headers[99]
        self.FiberRefStart = self.Headers[34] + 1
        self.FiberRefStop = self.Headers[36] + 1
        self.Fiber_Position_Offset = self.Headers[28]
        self.MultiPpointPosMeter = self.Headers[51:61] * self.Spatial_Sampling_Meters + self.Fiber_Position_Offset

    def get_all(self):
        # Headers, Fiber_Position_Offset, Trigger_Frequency, FiberRefStart,
        # FiberRefStop, MultiPpointPosMeter, RefUpdate_Window
        return [self.Headers, self.Fiber_Position_Offset, self.Trigger_Frequency,
                self.FiberRefStart, self.FiberRefStop,
                self.MultiPpointPosMeter, self.RefUpdate_Window]


def load_header(fHeaderSize: int, Headers: list, Spatial_Sampling_Meters: int) -> list:
    """

    :parameter fHeaderSize: tamaño del header del archivo
    :parameter Headers: header del archivo
    :parameter Spatial_Sampling_Meters: nyestra es

    :return:
    """
    if fHeaderSize > 101:
        if Headers[99] == 0:
            Headers[99] = 1
        if Headers[100] == 0:
            Headers[100] = 1
        if Headers[102] == 0:
            Headers[102] = 2
    else:
        pass

    foptions = fileTypeOptions(Headers, Spatial_Sampling_Meters)

    switcher = {'0': lambda: foptions.strain_raw_data(),
                '1': lambda: foptions.temp_raw_data(),
                '2': lambda: foptions.strain_map(),
                '3': lambda: foptions.temp_map()}

    mode = switcher.get(str(int(Headers[101])))

    if not mode:
        raise print('Code Header [101] not valid')
    else:
        mode()

    return foptions.get_all()


def readHeader(fullPath: str) -> np.array:
    """

    :parameter fullPath:  lugar del archivo bin

    :return: devuelve la header del archivo
    """
    fileID = open(fullPath, "rb")
    fileID.seek(0)  # It is unecessary but I'm more comfortable if I put it here.
    dt = np.dtype('<f8')
    headersize = np.fromfile(fileID, dtype=dt, count=1)
    fHeaderSize = int(headersize[0])
    header_data = np.fromfile(fileID, dtype=dt, count=(fHeaderSize-1))
    Headers = np.hstack((headersize,  header_data))
    fileID.close()
    return Headers


def LoadData(fullPath: str) -> list:
    """

    :parameter fullPath: ruta absoluta donde se encuentra el archivo bin
    :parameter RawData: indica si el tipo de dato es RawData o no

    :return:
    """
    fileID = open(fullPath, "rb")
    fileID.seek(0)
    dt = np.dtype('<f8')
    headersize = np.fromfile(fileID, dtype=dt, count=1)
    fHeaderSize = int(headersize[0])
    header_data = np.fromfile(fileID, dtype=dt, count=(fHeaderSize-1))
    if header_data[100] == 0 or header_data[100] == 1:
        RawData = True
        LOGGER.info('Archivo tipo RawData')

    else:
        RawData = False

    dt = np.int16 if RawData else np.float

    raw_file = np.fromfile(fileID, dtype=dt)
    fileID.close()

    # Get the FileHeader; The first Position is the FileHeader Size
    Headers = np.hstack((headersize,  header_data))
    N_processed_Points = int(Headers[14] - Headers[12])

    if fHeaderSize > 101:
        if Headers[99] == 0:
            Headers[99] = 1
        if Headers[100] == 0:
            Headers[100] = 1
        if Headers[102] == 0:
            Headers[102] = 2
    else:
        pass

    N_Time_Samples = int(len(raw_file) / N_processed_Points)

    # assert (len(raw_file) % N_processed_Points) != 0, LOGGER.error('The dimension are incorrect! ')
    if (len(raw_file) % N_processed_Points) != 0:
        LOGGER.error('The dimension are incorrect! ')
        sys.exit()
    try:
        TracesMatrix = raw_file.reshape((N_Time_Samples, N_processed_Points))
        TracesMatrix = TracesMatrix.transpose()
    except ValueError:
        LOGGER.error('Error loading the matrix')
        TracesMatrix = None
    del raw_file

    # TODO: CUIDADO
    if Headers[101] !=0 and Headers[101]!=1:
        TracesMatrix = TracesMatrix * 10            # mKevin to nStrain

    LOGGER.debug('[+] The matrix is loaded')

    return [TracesMatrix, Headers, N_processed_Points, N_Time_Samples, RawData]
