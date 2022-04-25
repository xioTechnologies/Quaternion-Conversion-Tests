import imufusion
import numpy
from scipy.spatial.transform import Rotation as Rotation

# --------------------------------------------------------------------------------
# Generate quaternions

axes = numpy.array([[-1, -1, -1], [-1, -1, 0], [-1, -1, 1],
                    [-1, 0, -1], [-1, 0, 0], [-1, 0, 1],
                    [-1, 1, -1], [-1, 1, 0], [-1, 1, 1],
                    [0, -1, -1], [0, -1, 0], [0, -1, 1],
                    [0, 0, -1], [0, 0, 1],
                    [0, 1, -1], [0, 1, 0], [0, 1, 1],
                    [1, -1, -1], [1, -1, 0], [1, -1, 1],
                    [1, 0, -1], [1, 0, 0], [1, 0, 1],
                    [1, 1, -1], [1, 1, 0], [1, 1, 1]])

angles = [-135, 45, 45, 135]  # do not include 90 degrees or 180 degrees to avoid gimbal lock

quaternions = numpy.array([[1, 0, 0, 0]])

for axis in axes:
    axis = axis / numpy.linalg.norm(axis)

    for angle in angles:
        angle = numpy.radians(angle)

        w = numpy.cos(angle/2)
        x = axis[0] * numpy.sin(angle/2)
        y = axis[1] * numpy.sin(angle/2)
        z = axis[2] * numpy.sin(angle/2)

        q = [w, x, y, z]  # https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToQuaternion/index.htm

        quaternions = numpy.append(numpy.array(quaternions), numpy.array([q]), axis=0)

numpy.savetxt("matlab_quaternions.csv", quaternions, delimiter=",")

# --------------------------------------------------------------------------------
# Matrix conversion functions


def to_matrix_euclideanspace(q):
    # https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm

    m11 = 1 - 2 * q[2] * q[2] - 2 * q[3] * q[3]
    m12 = 2 * q[1] * q[2] - 2 * q[3] * q[0]
    m13 = 2 * q[1] * q[3] + 2 * q[2] * q[0]
    m21 = 2 * q[1] * q[2] + 2 * q[3] * q[0]
    m22 = 1 - 2 * q[1] * q[1] - 2 * q[3] * q[3]
    m23 = 2 * q[2] * q[3] - 2 * q[1] * q[0]
    m31 = 2 * q[1] * q[3] - 2 * q[2] * q[0]
    m32 = 2 * q[2] * q[3] + 2 * q[1] * q[0]
    m33 = 1 - 2 * q[1] * q[1] - 2 * q[2] * q[2]

    return numpy.array([m11, m12, m13, m21, m22, m23, m31, m32, m33])


def to_matrix_fusion_installed(q):
    # https://pypi.org/project/imufusion

    return imufusion.Quaternion(numpy.array([q[0], q[1], q[2], q[3]])).to_matrix().flatten()


def to_matrix_fusion_v1_0_1(q):
    # https://github.com/xioTechnologies/Fusion/releases/tag/v1.0.1

    qwqw = q[0] * q[0]
    qwqx = q[0] * q[1]
    qwqy = q[0] * q[2]
    qwqz = q[0] * q[3]
    qxqy = q[1] * q[2]
    qxqz = q[1] * q[3]
    qyqz = q[2] * q[3]

    xx = 2 * (qwqw - 0.5 + q[1] * q[1])
    xy = 2 * (qxqy + qwqz)
    xz = 2 * (qxqz - qwqy)
    yx = 2 * (qxqy - qwqz)
    yy = 2 * (qwqw - 0.5 + q[2] * q[2])
    yz = 2 * (qyqz + qwqx)
    zx = 2 * (qxqz + qwqy)
    zy = 2 * (qyqz - qwqx)
    zz = 2 * (qwqw - 0.5 + q[3] * q[3])

    return numpy.array([xx, xy, xz, yx, yy, yz, zx, zy, zz])
    # return numpy.array([xx, yx, zx, xy, yy, zy, xz, yz, zz])  # transpose matrix


def to_matrix_kuipers(q):
    # page 168 of Quaternions and Rotation Sequence by Jack B. Kuipers, ISBN 0-691-10298-8

    m11 = 2 * q[0] * q[0] - 1 + 2 * q[1] * q[1]
    m12 = 2 * q[1] * q[2] + 2 * q[0] * q[3]
    m13 = 2 * q[1] * q[3] - 2 * q[0] * q[2]
    m21 = 2 * q[1] * q[2] - 2 * q[0] * q[3]
    m22 = 2 * q[0] * q[0] - 1 + 2 * q[2] * q[2]
    m23 = 2 * q[2] * q[3] + 2 * q[0] * q[1]
    m31 = 2 * q[1] * q[3] + 2 * q[0] * q[2]
    m32 = 2 * q[2] * q[3] - 2 * q[0] * q[1]
    m33 = 2 * q[0] * q[0] - 1 + 2 * q[3] * q[3]

    return numpy.array([m11, m12, m13, m21, m22, m23, m31, m32, m33])
    # return numpy.array([m11, m21, m31, m12, m22, m32, m13, m23, m33])  # transpose matrix


def find_in_csv(q, file_name):
    decimals = 3

    q = numpy.around(q, decimals)

    for row in numpy.genfromtxt(file_name, delimiter=","):
        if numpy.array_equal(q, numpy.around(row[0:4], decimals)):
            return row[4:]

    raise Exception("Cannot find quaternion " + str(q) + " in " + file_name)


def to_matrix_matlab(q):
    # https://www.mathworks.com/help/robotics/ref/quat2rotm.html

    return find_in_csv(q, "matlab_matrix.csv")


def to_matrix_scipy(q):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html

    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix().flatten()


def to_matrix_wikipedia(q):
    # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Rotation_matrices

    m11 = 1 - 2 * (q[2] * q[2] + q[3] * q[3])
    m12 = 2 * (q[1] * q[2] - q[0] * q[3])
    m13 = 2 * (q[0] * q[2] + q[1] * q[3])
    m21 = 2 * (q[1] * q[2] + q[0] * q[3])
    m22 = 1 - 2 * (q[1] * q[1] + q[3] * q[3])
    m23 = 2 * (q[2] * q[3] - q[0] * q[1])
    m31 = 2 * (q[1] * q[3] - q[0] * q[2])
    m32 = 2 * (q[0] * q[1] + q[2] * q[3])
    m33 = 1 - 2 * (q[1] * q[1] + q[2] * q[2])

    return numpy.array([m11, m12, m13, m21, m22, m23, m31, m32, m33])

# --------------------------------------------------------------------------------
# Euler conversion functions


def to_euler_euclideanspace(q):
    # https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler/index.htm

    heading = numpy.arctan2(2 * q[2] * q[0] - 2 * q[1] * q[3], 1 - 2 * q[2] * q[2] - 2 * q[3] * q[3])
    attitude = numpy.arcsin(2 * q[1] * q[2] + 2 * q[3] * q[0])
    bank = numpy.arctan2(2 * q[1] * q[0] - 2 * q[2] * q[3], 1 - 2 * q[1] * q[1] - 2 * q[3] * q[3])

    return numpy.degrees(numpy.array([bank, attitude, heading]))


def to_euler_fusion_v1_0_1(q):
    # https://github.com/xioTechnologies/Fusion/releases/tag/v1.0.1

    qwqwMinusHalf = q[0] * q[0] - 0.5

    roll = numpy.degrees(numpy.arctan2(q[2] * q[3] - q[0] * q[1], qwqwMinusHalf + q[3] * q[3]))
    pitch = numpy.degrees(-1.0 * numpy.arcsin(2 * (q[1] * q[3] + q[0] * q[2])))
    yaw = numpy.degrees(numpy.arctan2(q[1] * q[2] - q[0] * q[3], qwqwMinusHalf + q[1] * q[1]))

    return numpy.array([roll, pitch, yaw])


def to_euler_fusion_installed(q):
    # https://pypi.org/project/imufusion

    return imufusion.Quaternion(numpy.array([q[0], q[1], q[2], q[3]])).to_euler()


def to_euler_kuipers(q):
    # page 168 of Quaternions and Rotation Sequence by Jack B. Kuipers, ISBN 0-691-10298-8

    m11 = 2 * q[0] * q[0] + 2 * q[1] * q[1] - 1
    m12 = 2 * q[1] * q[2] + 2 * q[0] * q[3]
    m13 = 2 * q[1] * q[3] - 2 * q[0] * q[2]
    m23 = 2 * q[2] * q[3] + 2 * q[0] * q[1]
    m33 = 2 * q[0] * q[0] + 2 * q[3] * q[3] - 1

    psi = numpy.arctan2(m12, m11)
    theta = numpy.arcsin(-m13)
    phi = numpy.arctan2(m23, m33)

    return numpy.degrees(numpy.array([phi, theta, psi]))


def to_euler_matlab(q):
    # https://www.mathworks.com/help/robotics/ref/quat2eul.html

    return find_in_csv(q, "matlab_euler.csv")


def to_euler_scipy(q):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html

    rotation = Rotation.from_quat([q[1], q[2], q[3], q[0]])

    euler = rotation.as_euler("zyx", degrees=True)

    return numpy.array([euler[2], euler[1], euler[0]])


def to_euler_wikipedia(q):
    # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    phi = numpy.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] * q[1] + q[2] * q[2]))
    theta = numpy.arcsin(2 * (q[0] * q[2] - q[3] * q[1]))
    psi = numpy.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]))

    return numpy.degrees(numpy.array([phi, theta, psi]))

# --------------------------------------------------------------------------------
# Test conversions


def test(functions, compare):
    outputs = []

    for function in functions:
        print(function.__name__ + " ", end="")

        outputs.append([function(quaternion) for quaternion in quaternions])

    print("")

    for row_outputs in outputs:
        for column_index, column_outputs in enumerate(outputs):
            failed = 0

            for index, _ in enumerate(column_outputs):
                if compare(row_outputs[index], column_outputs[index]):
                    failed += 1

            if failed > 0:
                result = str(failed) + " failed"
            else:
                result = "Passed"

            print(result.ljust(len(functions[column_index].__name__)) + " ", end="")

        print("")


to_matrix_functions = [to_matrix_euclideanspace,
                       to_matrix_fusion_v1_0_1,
                       to_matrix_fusion_installed,
                       to_matrix_kuipers,
                       to_matrix_matlab,
                       to_matrix_scipy,
                       to_matrix_wikipedia]


def compare_matrix(a, b):
    return numpy.sum(numpy.absolute(numpy.around(a - b, 3))) != 0


test(to_matrix_functions, compare_matrix)

print("")

to_euler_functions = [to_euler_euclideanspace,
                      to_euler_fusion_v1_0_1,
                      to_euler_fusion_installed,
                      to_euler_kuipers,
                      to_euler_matlab,
                      to_euler_scipy,
                      to_euler_wikipedia]


def compare_euler(a, b):
    return numpy.sum(numpy.absolute((numpy.around(a, 0) % 360) - (numpy.around(b, 0) % 360))) != 0


test(to_euler_functions, compare_euler)
