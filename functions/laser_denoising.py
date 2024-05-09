import numpy as np
from functions import LOGGER

PreviousStrain = None


def vectorRefFiber(FiberRefStop, FiberRefStart):
    Sqrt_Ref_Fibre_Length = np.floor(np.sqrt(FiberRefStop - FiberRefStart + 1))
    segtotal = np.arange(FiberRefStart - 1, FiberRefStop)

    return int(Sqrt_Ref_Fibre_Length), segtotal.astype(int).transpose()


def laserDenoisingRefFiberv2(data: object, FiberRefStop: int,
                             FiberRefStart: int, RawData,
                             aditional: list = None) -> (np.array, list):
    """
    Laser Denoising Using Reference Fiber (2nd order correction, if Ref Update Denoise is activated)

    :param RawData: indica si el dato es de tipo Rawdata o no.
    :parameter data: datos de entrada
    :parameter FiberRefStop: Referencia final para hacer el láser denoising
    :parameter FiberRefStart: Referencia inicial para hacer el láser denoising
    :parameter aditional: parametros de strain reference y previous strain para hacer
        el laser denoising

    :return:
    """
    global PreviousStrain
    if not RawData:
        return data.Data, [], []
    else:
        LOGGER.debug('[+] Doing laser denoising...')

        THRESHOLD = 12000
        SUBTRACT_JUMP = 20000
        DIVIDE_VARIATION = 400
        Data = data.Data

        Sqrt_Ref_Fibre_Length, segtotal = vectorRefFiber(FiberRefStop, FiberRefStart)

        index_variations = np.greater(Data, THRESHOLD)
        variations = np.add(np.multiply(np.subtract(Data, SUBTRACT_JUMP), index_variations),
                            np.multiply(Data, ~index_variations))
        variation_matrix = np.divide(variations, DIVIDE_VARIATION)
        res = np.multiply(variation_matrix, index_variations)

        if aditional is not None:
            StrainReference, PreviousStrain = aditional.copy()
            res = np.hstack([StrainReference.reshape((len(StrainReference), 1)), res])
        strain_reference_matrix = np.cumsum(res, axis=1)

        if aditional is not None:
            strain_reference_matrix = strain_reference_matrix[:, 1:]
            res = res[:, 1:]
        Data = np.subtract(np.add(strain_reference_matrix, variation_matrix), res)

        if aditional is None:
            PreviousStrain = Data[:, 0]

        def get_segmented_mean(x):
            global PreviousStrain
            segmented_mean = np.mean(x[segtotal])
            val = np.subtract(x, segmented_mean)
            PreviousStrain = val.copy()
            return val


        laserRef = np.mean(Data[segtotal, :], 0)
        Data2 = np.apply_along_axis(get_segmented_mean, 0, Data)

        aditional_out = [strain_reference_matrix[:, -1].copy(), PreviousStrain.copy()]

        return np.multiply(Data2, data.ScaleFactor), np.multiply(laserRef, data.ScaleFactor), aditional_out
