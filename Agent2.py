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
from PIL import Image, ImageChops, ImageFilter

def equal(figureOne, figureTwo):

    img1 = figureOne.convert("RGB")
    data1 = img1.getdata()
    img2 = figureTwo.convert("RGB")
    data2 = img2.getdata()
    totalblack1 = 0.0
    totalblack2 = 0.0
    intersection = 0.0

    for item1, item2 in zip(data1,data2):
        if item1[0] == 0 and item1[1] == 0 and item1[2] == 0:
            totalblack1 += 1.0
        if item2[0] == 0 and item2[1] == 0 and item2[2] == 0:
            totalblack2 += 1.0
        if item1[0] == item2[0] and item1[1] == item2[1]  and item1[2] == item2[2]:
            if item1[0] == 0 and item1[1] == 0 and item1[2] == 0:
                intersection += 1.0
    if(totalblack1 - intersection <= 500 and totalblack2 - intersection <= 500):
        return True
    else : return False or ImageChops.difference(figureOne, figureTwo).getbbox() is None

def equalForDifference(figureOne, figureTwo):

    img1 = figureOne.convert("RGB")
    data1 = img1.getdata()
    img2 = figureTwo.convert("RGB")
    data2 = img2.getdata()
    totalblack1 = 0.0
    totalblack2 = 0.0
    intersection = 0.0

    for item1, item2 in zip(data1,data2):
        #print 'item1, item2 = ', item1, item2
        if item1[0] <= 128 and item1[1] <= 128 and item1[2] <= 128:
            totalblack1 += 1.0
        if item2[0] <= 128 and item2[1] <= 128 and item2[2] <= 128:
            totalblack2 += 1.0
        if (abs(item1[0] - item2[0]) < 128 and abs(item1[1] - item2[1]) < 128  and abs(item1[2] - item2[2]) < 128):
            intersection += 1.0

    #print 'totalblack1, totalblack2, intersection', totalblack1, totalblack2, intersection

    if(abs(totalblack1 - intersection) <= 300 and abs(totalblack2 - intersection) <= 300):
        return True
    else : return False or ImageChops.difference(figureOne, figureTwo).getbbox() is None


def darkPixelRatio(figureA):
    total = 0.00
    white = 0.00

    img = figureA.convert("RGB")
    datas = img.getdata()
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            white += 1.0
        total += 1.0
    return (float)((total - white)/total)*100

def darkPixelRatioCompare(figureA, figureB):
    
    dark1 = darkPixelRatio(figureA)
    dark2 = darkPixelRatio(figureB)

    return abs(dark1-dark2)

def darkPixel(figure):
    total = 0.00
    black = 0.00

    img = figure.convert("RGB")
    datas = img.getdata()
    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            black += 1.0
        total += 1.0
    return black   

def intersectionPixelRatio(figureOne, figureTwo):
    figureOneImage = Image.open(figureOne.visualFilename)
    figureTwoImage = Image.open(figureTwo.visualFilename)

    img1 = figureOneImage.convert("RGB")
    data1 = img1.getdata()
    img2 = figureTwoImage.convert("RGB")
    data2 = img2.getdata()
    totalblack1 = 0.0
    totalblack2 = 0.0
    intersection = 0.0

    for item1, item2 in zip(data1,data2):
        if item1[0] == 0 and item1[1] == 0 and item1[2] == 0:
            totalblack1 += 1.0
        if item2[0] == 0 and item2[1] == 0 and item2[2] == 0:
            totalblack2 += 1.0
        if item1[0] == item2[0] and item1[1] == item2[1]  and item1[2] == item2[2]:
            if item1[0] == 0 and item1[1] == 0 and item1[2] == 0:
                intersection += 1.0

    return (float)(intersection/totalblack1)*100.0 , (float)(intersection/totalblack2)*100.0

def checkAffineTransforms(figureOne, figureTwo):
    figureOneImage = Image.open(figureOne.visualFilename)
    figureTwoImage = Image.open(figureTwo.visualFilename)
    if(equal(figureOneImage, figureTwoImage)):
        return 0
    else :
        return checkRotation(figureOneImage, figureTwoImage)

def checkRotation(figureOne, figureTwo):
    size = 48, 48
    if (equal(figureOne, figureTwo.transpose(Image.FLIP_LEFT_RIGHT))):
        return 1
    elif (equal(figureOne, figureTwo.transpose(Image.FLIP_TOP_BOTTOM))):
        return 2
    elif (equal(figureOne, figureTwo.rotate(45))):
        return 3
    elif (equal(figureOne, figureTwo.rotate(90))):
        return 4
    elif (equal(figureOne, figureTwo.rotate(135))):
        return 5
    elif (equal(figureOne, figureTwo.rotate(180))):
        return 6
    elif (equal(figureOne, figureTwo.rotate(270))):
        return 7
    else :
        return -1

def tversky(figureA, figureB):
    figureOneLoaded = figureA.load()
    figureTwoLoaded = figureB.load()
    andAB = 0
    diffAB = 0
    diffBA = 0

    for i in range(0, figureA.size[0]):
        for j in range(0, figureA.size[1]):
            onePixel = figureOneLoaded[i, j]
            twoPixel = figureTwoLoaded[i, j]
            if(onePixel == twoPixel):
                andAB += 1
            elif(onePixel > twoPixel):
                diffAB += 1
            else :
                diffBA += 1

    tversky = 1.0 *(andAB/(andAB + 0.5*diffAB + 0.5*diffBA))
    return tversky

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

        print 'problem: ', problem.name
        ######################################################################## 
        ######################################################################## 
        ############################## 2X2 #####################################
        ######################################################################## 
        ########################################################################

        if(problem.problemType == "2x2"):
            figureA = problem.figures['A']
            figureB = problem.figures['B']
            figureC = problem.figures['C']

            figureOneImage = Image.open(figureA.visualFilename)
            figureTwoImage = Image.open(figureB.visualFilename)
            figureThreeImage = Image.open(figureC.visualFilename)

            #Convert to format "1"
            figureAImage = figureOneImage.convert("1")
            figureBImage = figureTwoImage.convert("1")
            figureCImage = figureThreeImage.convert("1")

            confidence = [0.0 , 0.0, 0.0, 0.0, 0.0, 0.0]

            ####### Checking Affine Transformations
            transformValueAB = checkAffineTransforms(figureA, figureB)
            transformValueAC = checkAffineTransforms(figureA, figureC)

            for i in range(0,6):
                resultB = checkAffineTransforms(figureB, problem.figures[str(i+1)])
                resultC = checkAffineTransforms(figureC, problem.figures[str(i+1)])

                if(transformValueAB == resultC and transformValueAB != -1):
                    print 'affine AB, affine CD = ', transformValueAB, resultC
                    return i + 1
                elif (transformValueAC == resultB and transformValueAC != -1):
                    print 'affine AC, affine BD = ', transformValueAC, resultB
                    return i + 1

            ####### Difference between Images
            img1 = figureOneImage.filter(ImageFilter.FIND_EDGES)
            img2 = figureTwoImage.filter(ImageFilter.FIND_EDGES)
            img3 = figureThreeImage.filter(ImageFilter.FIND_EDGES)
            #img1.show()
            #img2.show()
            #img3.show()
            nextFig = None
            if(equalForDifference(img1,img2)):
                nextFig = img3
                diffA = ImageChops.difference(img1, img2)
            elif(equalForDifference(img1,img3)):
                nextFig = img2
                diffA = ImageChops.difference(img1, img3)
            if(nextFig != None):
                for i in range(0,6):
                    figureOptionImage = Image.open(problem.figures[str(i+1)].visualFilename)
                    imgAns = figureOptionImage.filter(ImageFilter.FIND_EDGES)
                    result = ImageChops.difference(nextFig, imgAns)
                    if(equalForDifference(diffA,result)):
                        print 'returning equalForDifference'
                        return i + 1

            ########################## Dark Pixel count ######################
            darkAB = darkPixel(figureAImage)/(1 + darkPixel(figureBImage))
            darkAC = darkPixel(figureAImage)/(1 + darkPixel(figureCImage))

            for i in range(0,6):
                figureOptionImage = Image.open(problem.figures[str(i+1)].visualFilename)
                figureOptionImage = figureOptionImage.convert("1")
                darkAnsB = darkPixel(figureBImage)/(1 + darkPixel(figureOptionImage))
                darkAnsC = darkPixel(figureCImage)/(1 + darkPixel(figureOptionImage))
                if(abs(darkAB - darkAnsC) < 0.05):
                    confidence[i] += 1/(1 + (abs(darkAB - darkAnsC)*abs(darkAB - darkAnsC)))
                if(abs(darkAC - darkAnsB) < 0.05):
                    confidence[i] += 1/(1 + (abs(darkAC - darkAnsB)*abs(darkAC - darkAnsB)))

            print 'confidence = ', confidence
            ####### OR between Images

            andAB = ImageChops.logical_and(figureAImage, figureBImage)
            andAC = ImageChops.logical_and(figureAImage, figureCImage) 

            for i in range(0,6):
                figureOptionImage = Image.open(problem.figures[str(i+1)].visualFilename)
                figureOptionImage = figureOptionImage.convert("1")
                andCAns = ImageChops.logical_and(figureCImage, figureOptionImage)
                andBAns = ImageChops.logical_and(figureBImage, figureOptionImage)
                if(tversky(andAB, andCAns) > 0.97):
                    confidence[i] += tversky(andAB, andCAns)
                if(tversky(andAC, andBAns) > 0.97):
                    confidence[i] += tversky(andAC, andBAns)

            print 'confidence = ', confidence
            ####### AND between Images

            orAB = ImageChops.logical_or(figureAImage, figureBImage)
            orAC = ImageChops.logical_or(figureAImage, figureCImage) 

            for i in range(0,6):
                figureOptionImage = Image.open(problem.figures[str(i+1)].visualFilename)
                figureOptionImage = figureOptionImage.convert("1")
                orCAns = ImageChops.logical_or(figureCImage, figureOptionImage)
                orBAns = ImageChops.logical_or(figureBImage, figureOptionImage)
                if(tversky(orAB, orCAns) > 0.97):
                    confidence[i] += tversky(orAB, orCAns)
                if(tversky(orAC, orBAns) > 0.97):
                    confidence[i] += tversky(orAC, orBAns)
            print 'confidence = ', confidence
            ####### XOR between Images

            xorAB = ImageChops.logical_xor(figureAImage, figureBImage)
            xorAC = ImageChops.logical_xor(figureAImage, figureCImage) 

            for i in range(0,6):
                figureOptionImage = Image.open(problem.figures[str(i+1)].visualFilename)
                figureOptionImage = figureOptionImage.convert("1")
                xorCAns = ImageChops.logical_xor(figureCImage, figureOptionImage)
                xorBAns = ImageChops.logical_xor(figureBImage, figureOptionImage)
                if(tversky(xorAB, xorCAns) > 0.97):
                    confidence[i] += tversky(xorAB, xorCAns)
                if(tversky(xorAC, xorBAns) > 0.97):
                    confidence[i] += tversky(xorAC, xorBAns)

            maximum = 0.0
            result_index = -1
            print 'confidence = ', confidence
            for i in range(0,6):
                if(confidence[i] > maximum):
                    maximum = confidence[i]
                    result_index = i + 1

            return result_index

        ######################################################################## 
        ######################################################################## 
        ############################## 3X3 #####################################
        ######################################################################## 
        ######################################################################## 

        if(problem.problemType == "3x3"):

            confidence = [0.0 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            figureA = problem.figures['A']
            figureB = problem.figures['B']
            figureC = problem.figures['C']
            figureD = problem.figures['D']
            figureE = problem.figures['E']
            figureF = problem.figures['F']
            figureG = problem.figures['G']
            figureH = problem.figures['H']

            figureAImage = Image.open(figureA.visualFilename)
            figureBImage = Image.open(figureB.visualFilename)
            figureCImage = Image.open(figureC.visualFilename)
            figureDImage = Image.open(figureD.visualFilename)
            figureEImage = Image.open(figureE.visualFilename)
            figureFImage = Image.open(figureF.visualFilename)
            figureGImage = Image.open(figureG.visualFilename)
            figureHImage = Image.open(figureH.visualFilename)

            #Convert to format "1"
            figureAImage = figureAImage.convert("1")
            figureBImage = figureBImage.convert("1")
            figureCImage = figureCImage.convert("1")
            figureDImage = figureDImage.convert("1")
            figureEImage = figureEImage.convert("1")
            figureFImage = figureFImage.convert("1")
            figureGImage = figureGImage.convert("1")
            figureHImage = figureHImage.convert("1")

            ######################################################################## 

                        #######################Horizontal Relation Check####################
            transformAB = checkAffineTransforms(figureA, figureB)
            transformBC = checkAffineTransforms(figureB, figureC)

            transformDE = checkAffineTransforms(figureD, figureE)
            transformEF = checkAffineTransforms(figureE, figureF)

            transformGH = checkAffineTransforms(figureG, figureH)

            h1 = -1
            h2 = -1

            if(transformAB == transformBC):
                h1 = transformAB

            if(transformDE == transformEF):
                h2 = transformDE

            if(h1 == h2 and h1 != -1):
                for i in range(0,8):
                    resultH = checkAffineTransforms(figureH, problem.figures[str(i+1)])
                    if(transformGH == resultH and transformGH != -1):
                        return i + 1

            ######################################################################## 

            ####################### Vertical Relation Check ####################
            transformAD = checkAffineTransforms(figureA, figureD)
            transformDG = checkAffineTransforms(figureD, figureG)

            transformBE = checkAffineTransforms(figureB, figureE)
            transformEH = checkAffineTransforms(figureE, figureH)

            transformCF = checkAffineTransforms(figureC, figureF)

            v1 = -1
            v2 = -1

            if(transformAD == transformDG):
                v1 = transformAD

            if(transformBE == transformEH):
                v2 = transformBE

            if(v1 == v2 and v1 != -1):
                for i in range(0,8):
                    resultF = checkAffineTransforms(figureF, problem.figures[str(i+1)])
                    if(transformCF == resultF and transformCF != -1):
                        return i + 1

            ######################################################################## 

            ########################## Dark Pixel count ######################
            darkAB = darkPixel(figureAImage)/darkPixel(figureBImage)
            darkBC = darkPixel(figureBImage)/darkPixel(figureCImage)
            darkDE = darkPixel(figureDImage)/darkPixel(figureEImage) 
            darkEF = darkPixel(figureEImage)/darkPixel(figureFImage) 
            darkGH = darkPixel(figureGImage)/darkPixel(figureHImage) 

            ABC =  darkAB/darkBC
            DEF = darkDE/darkEF 

            if(abs(ABC - DEF) < 0.05):
                for i in range(0,8):
                    figureOptionImage = Image.open(problem.figures[str(i+1)].visualFilename)
                    figureOptionImage = figureOptionImage.convert("1")
                    darkAns = darkPixel(figureHImage)/darkPixel(figureOptionImage)
                    GHAns = darkGH/darkAns
                    if(abs(ABC - GHAns) < 0.05 and abs(DEF - GHAns) < 0.05):
                        confidence[i] += 1/(1 + (abs(ABC - GHAns) + abs(DEF - GHAns)*abs(ABC - GHAns) + abs(DEF - GHAns)))


            ######################################################################## 

            ############################# AND ############################

            outAC = ImageChops.logical_or(figureAImage, figureCImage)
            outDF = ImageChops.logical_or(figureDImage, figureFImage)

            outAG = ImageChops.logical_or(figureAImage, figureGImage)
            outBH = ImageChops.logical_or(figureBImage, figureHImage)

            flag = False
            if(equal(figureAImage, outAC)):
                if(equal(figureDImage, outDF)):
                    for i in range(0,8):
                        figureOptionImage = Image.open(problem.figures[str(i+1)].visualFilename)
                        figureOptionImage = figureOptionImage.convert("1")
                        outAns = ImageChops.logical_or(figureGImage, figureOptionImage)
                        if(equal(figureGImage, outAns)):
                            flag = True
                            confidence[i] += tversky(figureGImage, outAns)

            if(equal(figureAImage, outAG)):
                if(equal(figureBImage, outBH)):
                    for i in range(0,8):
                        figureOptionImage = Image.open(problem.figures[str(i+1)].visualFilename)
                        figureOptionImage = figureOptionImage.convert("1")
                        outAns = ImageChops.logical_or(figureCImage, figureOptionImage)
                        if(equal(figureCImage, outAns)):
                            flag = True
                            confidence[i] += tversky(figureCImage, outAns)

            ######################################################################## 

            ########################## XOR ###############################
            xorAB = ImageChops.logical_xor(figureAImage, figureBImage)
            xorBC = ImageChops.logical_xor(figureBImage, figureCImage)
            darkH1 = darkPixelRatioCompare(xorAB, xorBC)

            xorDE = ImageChops.logical_xor(figureDImage, figureEImage)
            xorEF = ImageChops.logical_xor(figureEImage, figureFImage)
            darkH2 = darkPixelRatioCompare(xorDE,xorEF)

            xorGH = ImageChops.logical_xor(figureGImage, figureHImage)


            xorAD = ImageChops.logical_xor(figureAImage, figureDImage)
            xorDG = ImageChops.logical_xor(figureDImage, figureGImage)
            darkV1 = darkPixelRatioCompare(xorAB, xorBC)

            xorBE = ImageChops.logical_xor(figureBImage, figureEImage)
            xorEH = ImageChops.logical_xor(figureEImage, figureHImage)
            darkV2 = darkPixelRatioCompare(xorDE,xorEF)

            xorCF = ImageChops.logical_xor(figureCImage, figureFImage)


            if(flag):
                for i in range(0,8):
                    figureOptionImage = Image.open(problem.figures[str(i+1)].visualFilename)
                    figureOptionImage = figureOptionImage.convert("1")
                    xorAns = ImageChops.logical_xor(figureHImage, figureOptionImage)
                    darkH3 = darkPixelRatioCompare(xorGH, xorAns) 
                    if(abs(darkH3 - darkH1) < 0.5 and abs(darkH3-darkH1) < 0.5):
                        confidence[i] += tversky(xorGH,xorAns)

            ######################################################################## 


            ########################## OR ###############################
            unionBD = ImageChops.logical_and(figureBImage, figureDImage)
            unionCE = ImageChops.logical_and(figureCImage, figureEImage)
            unionEG = ImageChops.logical_and(figureEImage, figureGImage)
            unionFH = ImageChops.logical_and(figureFImage, figureHImage)


            if(tversky(unionBD, figureEImage) > 0.97 and tversky(unionCE, figureFImage) > 0.97 and tversky(unionEG, figureHImage) > 0.97):
                for i in range(0,8):
                    figureOptionImage = Image.open(problem.figures[str(i+1)].visualFilename)
                    figureOptionImage = figureOptionImage.convert("1")
                    if(tversky(unionFH, figureOptionImage) > 0.97 ):
                        confidence[i] += tversky(unionFH, figureOptionImage) 

            ###########################################################################             


            ######################### Simple XOR ######################################
            XORFlag = False
            if(tversky(xorAB, xorDE) > 0.97):
                if(tversky(xorBC, xorEF) > 0.97):
                    for i in range(0,8):
                        figureOptionImage = Image.open(problem.figures[str(i+1)].visualFilename)
                        figureOptionImage = figureOptionImage.convert("1")
                        xorAns = ImageChops.logical_xor(figureHImage, figureOptionImage)
                        if(tversky(xorEF, xorAns) > 0.97):
                            XORFlag = True
                            confidence[i] += tversky(xorEF, xorAns)

            if(tversky(xorAD, xorBE) > 0.97):
                if(tversky(xorDG, xorEH) > 0.97):
                    for i in range(0,8):
                        figureOptionImage = Image.open(problem.figures[str(i+1)].visualFilename)
                        figureOptionImage = figureOptionImage.convert("1")
                        xorAns = ImageChops.logical_xor(figureFImage, figureOptionImage)
                        if(tversky(xorBE, xorCF) > 0.97 and tversky(xorEH, xorAns) > 0.97):
                            XORFlag = True
                            confidence[i] += tversky(xorEH, xorAns)
            ###########################################################################

            ################Affine Transformations with 2x1 Grid#####################

            boxright = 93 , 0, 184, 184
            boxleft = 0 , 0 , 92 , 184

            cropBRight = figureBImage.crop(boxright)
            cropERight = figureEImage.crop(boxright)

            cropHRight = figureHImage.crop(boxright)

            cropCLeft = figureCImage.crop(boxleft)
            cropFLeft = figureFImage.crop(boxleft)
            cropCRight = figureCImage.crop(boxright)
            cropFRight = figureFImage.crop(boxright)


            if(XORFlag and tversky(cropBRight, cropCLeft) < 0.97 and tversky(cropERight, cropFLeft) < 0.97 ):
                for i in range(0,8):
                    figureOptionImage = Image.open(problem.figures[str(i+1)].visualFilename)
                    figureOptionImage = figureOptionImage.convert("1")
                    cropAnsLeft = figureOptionImage.crop(boxleft)
                    cropAnsRight = figureOptionImage.crop(boxright)
                    confidence[i] += (tversky(cropHRight, cropAnsLeft)/(1 + darkPixelRatioCompare(cropHRight, cropAnsLeft) * darkPixelRatioCompare(cropHRight, cropAnsLeft)) 
                        + tversky(cropHRight, cropAnsRight)/(1 + darkPixelRatioCompare(cropHRight, cropAnsRight) * darkPixelRatioCompare(cropHRight, cropAnsRight)))

            ###########################################################################

            ##################Affine Transformations with 3x1 Grid#####################
            boxleft = 0 , 0 , 61 , 184
            boxmiddle = 61 , 0, 122, 184
            boxright = 122 , 0, 184, 184


            cropALeft = figureAImage.crop(boxleft)
            cropAMiddle = figureAImage.crop(boxmiddle)

            cropDLeft = figureDImage.crop(boxleft)
            cropDMiddle = figureDImage.crop(boxmiddle)

            cropGLeft = figureGImage.crop(boxleft)
            cropGMiddle = figureGImage.crop(boxmiddle)

            cropCRight = figureCImage.crop(boxright)
            cropCMiddle = figureCImage.crop(boxmiddle)

            cropFRight = figureFImage.crop(boxright)
            cropFMiddle = figureFImage.crop(boxmiddle)

            H1L = tversky(figureBImage.crop(boxmiddle),ImageChops.logical_and(cropALeft, cropAMiddle))
            H1R = tversky(figureBImage.crop(boxmiddle),ImageChops.logical_and(cropCRight, cropCMiddle))

            H2L = tversky(figureEImage.crop(boxmiddle),ImageChops.logical_and(cropDLeft, cropDMiddle))
            H2R = tversky(figureEImage.crop(boxmiddle),ImageChops.logical_and(cropFRight, cropFMiddle))

            if(abs(H1L - H1R) < 0.01 and abs(H2L - H2R) < 0.01):
                for i in range(0,8):
                    figureOptionImage = Image.open(problem.figures[str(i+1)].visualFilename)
                    figureOptionImage = figureOptionImage.convert("1")
                    cropAnsRight = figureOptionImage.crop(boxright)
                    cropAnsMiddle = figureOptionImage.crop(boxmiddle)
                    confidence[i] += tversky(figureHImage.crop(boxmiddle),ImageChops.logical_and(cropAnsRight, cropAnsMiddle))

            #########################################################################

            ################# Dark Pixel count between End Figures ###################
            darkAC = darkPixel(figureAImage)/darkPixel(figureCImage)
            darkDF = darkPixel(figureDImage)/darkPixel(figureFImage)

            if(abs(darkAC - darkDF) < 0.02):
                for i in range(0,8):
                    figureOptionImage = Image.open(problem.figures[str(i+1)].visualFilename)
                    figureOptionImage = figureOptionImage.convert("1")
                    darkAns = darkPixel(figureGImage)/darkPixel(figureOptionImage)
                    confidence[i] += 1/(1 + abs(darkAC + darkDF - 2*darkAns)*abs(darkAC + darkDF - 2*darkAns))

            #########################################################################

            maximum = 0.0
            result_index = -1
            for i in range(0,8):
                if(confidence[i] > maximum):
                    maximum = confidence[i]
                    result_index = i + 1

            if(result_index > -1):
                return result_index

        return -1
