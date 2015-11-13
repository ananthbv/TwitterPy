# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
#
# You may also create and submit new files in addition to modifying this file.
#
# Make sure your file retains methods with the signatures:
# def __init__(self)
# def Solve(self,problem)
#
# These methods will be necessary for the project's main method to run.

# Install Pillow and uncomment this line to access image processing.
from PIL import Image, ImageChops, ImageDraw, ImageFilter
import math, operator

Images = {}
rgbImages = {}
grayscaleImages = {}
Images1 = {}

SIMILARITY_THRESHOLD = 500
solutionImageList = []

def checkForEquality(i1, i2):
    if exactlyEqual(i1, i2):
        #print 'images ', i1, ' and', i2, ' exactly equal.'
        return True
    elif almostEqual(i1, i2):
        # diff = rmsdiff(Images['A'], Images['B'])  # rms diff does not work - returns 0
                                                    # even for images rotated/reflected
        #print 'images ', i1, ' and', i2, ' almost equal.'
        return True
    return False

# Taken from: http://effbot.org/zone/pil-comparing-images.htm
def exactlyEqual(i1, i2):
    if isinstance(i1, basestring):
        img1 = rgbImages[i1]
    else:
        img1 = i1

    if isinstance(i2, basestring):
        img2 = rgbImages[i2]
    else:
        img2 = i2

    diff = ImageChops.difference(img1, img2).getbbox()
    if diff is None:
        return True
    else:
        #print 'diff = ', diff
        return False


def almostEqual(i1, i2):
    if isinstance(i1, basestring):
        img1 = rgbImages[i1]
    else:
        img1 = i1

    if isinstance(i2, basestring):
        img2 = rgbImages[i2]
    else:
        img2 = i2

    img1Data = img1.getdata()
    img2Data = img2.getdata()
    img1BlackCount = 0
    img2BlackCount = 0
    same = 0
    #print 'comparing for almost equal', i1, i2
    img1colors = img1.getcolors()
    img2colors = img2.getcolors()
    for pixelCount, rgb in img1colors:
        if rgb == (0, 0, 0):
            img1BlackCount = pixelCount
    for pixelCount, rgb in img2colors:
        if rgb == (0, 0, 0):
            img2BlackCount = pixelCount

    for pix1, pix2 in zip(img1Data, img2Data):
        #print 'pix1, pix2 = ', pix1, pix2
        if pix1 == (0, 0, 0):
            if pix1 == pix2:
                same += 1
    print 'img1BlackCount , img2BlackCount, same = ', img1BlackCount, img2BlackCount, same
    if abs(img1BlackCount - img2BlackCount) <= SIMILARITY_THRESHOLD and \
                    img1BlackCount - same <= SIMILARITY_THRESHOLD and \
                            img2BlackCount - same <= SIMILARITY_THRESHOLD:

        return True

    if float(same)/float(img1BlackCount) > 0.95 and float(same)/float(img2BlackCount) > 0.95:
        return True

    return False

def checkForFill(i1, i2):
    img1 = rgbImages[i1].filter(ImageFilter.FIND_EDGES)
    img2 = rgbImages[i2].filter(ImageFilter.FIND_EDGES)

    return checkForEquality(img1, img2)

def checkForReflection(i1, i2, reflectionType, problem):
    img1 = rgbImages[i1]
    img2 = rgbImages[i2]
    reflected = None

    if reflectionType == 'h':
        reflected = img2.transpose(Image.FLIP_LEFT_RIGHT)

    if reflectionType == 'v':
        reflected = img2.transpose(Image.FLIP_TOP_BOTTOM)
    return checkForEquality(img1, reflected)

# Taken from http://snipplr.com/view/757/
# Works
def rmsdiff(im1, im2):
    h1 = im1.histogram()
    h2 = im2.histogram()

    rms = math.sqrt(reduce(operator.add, map(lambda a, b: (a - b) ** 2, h1, h2)) / len(h1))

    return rms

# Taken from: http://effbot.org/zone/pil-comparing-images.htm
# Does not work
def rmsdiff2(im1, im2):
    h = ImageChops.difference(im1, im2).histogram()
    print 'histogram h = ', h

    # calculate rms
    #return math.sqrt(sum(h*(i**2) for i, h in enumerate(h))) / (float(im1.size[0]) * im1.size[1]))
    return math.sqrt(reduce(operator.add,
        map(lambda h, i: h*(i**2), h, range(len(h)))
    ) / (float(im1.size[0]) * im1.size[1]))

def identifySolution(predictedSolution):
    solution = -1
    solutionImageList = ['1', '2', '3', '4', '5', '6']
    print 'solutions ', solutionImageList
    for i in solutionImageList:
        if checkForEquality(predictedSolution, i):
            solution = i

    print 'solution = ', solution
    return solution

def openAllImages(problem):
    if problem.problemType == '2x2':
        imageList = ['A', 'B', 'C', '1', '2', '3', '4', '5', '6']
        solutionImageList = ['1', '2', '3', '4', '5', '6']
    if problem.problemType == '3x3':
        imageList = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', '1', '2', '3', '4', '5', '6', '7', '8']
        solutionImageList = ['1', '2', '3', '4', '5', '6', '7', '8']

    for i in imageList:
        Images[i] = Image.open(problem.figures[i].visualFilename)
        rgbImages[i] = Images[i].convert('RGB')
        Images1[i] = Images[i].convert('1')
        grayscaleImages[i] = Images[i].convert('L')


class Agent:
    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().
    def __init__(self):
        pass

    # The primary method for solving incoming Raven's Progressive Matrices.
    # For each problem, your Agent's Solve() method will be called. At the
    # conclusion of Solve(), your Agent should return an integer representing its
    # answer to the question: "1", "2", "3", "4", "5", or "6". These integers
    # are also the Names of the individual RavensFigures, obtained through
    # RavensFigure.getName() (as Strings).
    #
    # In addition to returning your answer at the end of the method, your Agent
    # may also call problem.checkAnswer(int givenAnswer). The parameter
    # passed to checkAnswer should be your Agent's current guess for the
    # problem; checkAnswer will return the correct answer to the problem. This
    # allows your Agent to check its answer. Note, however, that after your
    # agent has called checkAnswer, it will *not* be able to change its answer.
    # checkAnswer is used to allow your Agent to learn from its incorrect
    # answers; however, your Agent cannot change the answer to a question it
    # has already answered.
    #
    # If your Agent calls checkAnswer during execution of Solve, the answer it
    # returns will be ignored; otherwise, the answer returned at the end of
    # Solve will be taken as your Agent's answer to this problem.
    #
    # Make sure to return your answer *as an integer* at the end of Solve().
    # Returning your answer as a string may cause your program to crash.
    def Solve(self,problem):

        solution = -1
        print 'problem: ', problem.name
        openAllImages(problem)
        # print ImageChops.difference(Images['A'], Images['B']).getbbox()
        #if problem.name != 'Basic Problem B-09':
        #    return -1

        predictedSolution = ''
        if problem.problemType == '2x2':
            print 'checking for equality of A and B'
            if checkForEquality('A', 'B'):  # Horizontal equality
                predictedSolution = 'C'
                solution = identifySolution(predictedSolution)
                if solution != -1:
                    return solution

            print 'checking for equality of A and C'
            if checkForEquality('A', 'C'):  # Vertical equality
                predictedSolution = 'B'
                solution = identifySolution(predictedSolution)
                if solution != -1:
                    return solution

            print 'checking for horizontal reflection of A and B'
            if checkForReflection('A', 'B', 'h', problem):  # Horizontal reflection
                print 'B is a horizontal reflection of A'
                predictedSolution = rgbImages['C'].transpose(Image.FLIP_LEFT_RIGHT)
                solution = identifySolution(predictedSolution)
                if solution != -1:
                    return solution

            print 'checking for horizontal reflection of A and C'
            if checkForReflection('A', 'C', 'h', problem):  # Horizontal reflection
                print 'C is a horizontal reflection of A'
                predictedSolution = rgbImages['B'].transpose(Image.FLIP_LEFT_RIGHT)
                solution = identifySolution(predictedSolution)
                if solution != -1:
                    return solution

            print 'checking for vertical reflection of A and C'
            if checkForReflection('A', 'C', 'v', problem):  # Vertical equality
                print 'C is a vertical reflection of A'
                predictedSolution = rgbImages['B'].transpose(Image.FLIP_TOP_BOTTOM)
                #if problem.name == 'Basic Problem B-06':
                    #predictedSolution.show()
                solution = identifySolution(predictedSolution)
                if solution != -1:
                    return solution

            print 'checking for fills between A and B'
            if checkForFill('A', 'B'):
                print 'B is a filled version of A'


        #ImageDraw.floodfill(Images['A'], xy, (0, 0, 0), border=None)
        mask = Images['A'].convert('L')
        th=150 # the value has to be adjusted for an image of interest
        mask = mask.point(lambda i: i < th and 255)
        #mask.show()
        #print 'mask = ', mask
        #for x in mask.getdata():
        #    print x

        #Images['B'].filter(ImageFilter.FIND_EDGES).show()
        #for x in Images['A'].convert('L').getdata():
        #    print x
        #print 'A histogram = ', Images1['A'].getdata()

        if problem.problemType == '3x3':
            if checkForEquality('A', 'B') and checkForEquality('B', 'C'):
                if checkForEquality('D', 'E') and checkForEquality('E', 'F'):
                    predictedSolution = 'H'
                    solution = identifySolution(predictedSolution)
                    if solution != -1:
                        return solution

        return solution
