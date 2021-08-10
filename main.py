import cv2 as cv
import numpy as np
import open3d as o3d
import datetime

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#---------------------------------------define-------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------


SADWindow=9
baseline=119.784
focal=1051.74462890625

cx = 944.3578491210938
cy = 536.3162841796875
fx = 1051.74462890625
fy = 1051.74462890625

path_img='D:\\AA_CourseFile\\2021SummerIntern\\opencv_learn\\code0804\\rectified\\'
path_depth='D:\\AA_CourseFile\\2021SummerIntern\\opencv_learn\\code0804\\depth\\'
path_glass='D:\\AA_CourseFile\\2021SummerIntern\\opencv_learn\\code0804\\glass\\'
path_mask='D:\\AA_CourseFile\\2021SummerIntern\\opencv_learn\\code0804\\mask\\'
path_cloud='D:\\AA_CourseFile\\2021SummerIntern\\opencv_learn\\code0804\\cloud\\'
path_depth_masked='D:\\AA_CourseFile\\2021SummerIntern\\opencv_learn\\code0804\\depth_masked\\'
path_cloud_glass='D:\\AA_CourseFile\\2021SummerIntern\\opencv_learn\\code0804\\cloud_glass\\'
path_depth_glass='D:\\AA_CourseFile\\2021SummerIntern\\opencv_learn\\code0804\\depth_glass\\'


#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#---------------------------------------depth--------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------


def create_depth(path_left,path_right,path_depth,write=True): #path of left and right img, path of depth
    left_image=cv.imread(path_left,cv.IMREAD_GRAYSCALE)
    right_image=cv.imread(path_right,cv.IMREAD_GRAYSCALE)

    stereo = cv.StereoSGBM_create(numDisparities=16*8, 
    blockSize=SADWindow, 
    P1=8*3*SADWindow*SADWindow, 
    P2=32*3*SADWindow*SADWindow,
    uniquenessRatio=10, 
    disp12MaxDiff=1, 
    speckleWindowSize=1,
    speckleRange=2,
    preFilterCap=63)

    depth = stereo.compute(left_image, right_image)

    depth_real=np.ones_like(depth, dtype=np.uint8)
    for a in range(0,depth.shape[0]):
        for b in range(0,depth.shape[1]):
            
            if depth[a][b]<1:
                depth_real[a][b]=0
            else:
                depth_real[a][b]=(baseline*focal)/depth[a][b]
                if depth_real[a][b]<32:
                    depth_real[a][b]=0

    if write==True:
        cv.imwrite(path_depth,depth_real)

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#---------------------------------------mask---------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

def create_mask(path_glass,path_mask,write=True):
    glass=cv.imread(path_glass,cv.IMREAD_GRAYSCALE)
    for x in range(glass.shape[0]):
        for y in range(glass.shape[1]):
            if glass[x][y]<128:
                glass[x][y]=0
            else:
                glass[x][y]=255
    mask_inv = cv.bitwise_not(glass)
    if write==True:
        cv.imwrite(path_mask,mask_inv)

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#---------------------------------------masked depth-------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

def create_masked_depth(path_depth,path_mask,path_depth_masked,write=True):
    depth=cv.imread(path_depth,0)
    mask=cv.imread(path_mask,0)
    masked=cv.bitwise_and(depth,depth,mask=mask)
    if write==True:
        cv.imwrite(path_depth_masked,masked)

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#---------------------------------------cloud--------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

def create_cloud(path_depth,path_img,path_cloud,write=True):
    depth=cv.imread(path_depth,0)
    color=cv.imread(path_img)
    color=cv.cvtColor(color,cv.COLOR_BGR2RGB)

    color = o3d.geometry.Image(color)
    depth = o3d.geometry.Image(depth)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(1920, 1080, fx, fy, cx, cy)

    pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    voxel_down_pcd = pc.uniform_down_sample(every_k_points=8)

    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=32,std_ratio=0.5)

    inlier_cloud = voxel_down_pcd.select_by_index(ind)

    if write==True:
        o3d.io.write_point_cloud(path_cloud, inlier_cloud)

def cloud_visualization(path_cloud):
    pcd = o3d.io.read_point_cloud(path_cloud)
    o3d.visualization.draw_geometries([pcd])

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#---------------------------------------create glass-------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------

def glass_contour(path_mask):
    mask=cv.imread(path_mask,0)
    (_,bin)=cv.threshold(mask,200,255,cv.THRESH_BINARY)

    (_,c,_) = cv.findContours(bin,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    c=sorted(c,key=cv.contourArea,reverse=True)

    '''
    for i in range(len(contours)):
        area=cv.contourArea(contours[i])
        print(area)
    '''
    contours=[]
    areaMax=cv.contourArea(c[0])/10
    for i in range(len(c)):
        area=cv.contourArea(c[i])
        if area>areaMax:
            contours.append(c[i])
    
    #draw=cv.drawContours(mask.copy(),contours,-1,(255,0,0),2)

    cnt=[]
    for i in range(len(contours)):
        cur_cnt=np.zeros((len(contours[i]),2))
        for j in range(len(contours[i])):
            cur_cnt[j]=contours[i][j][0]
        cnt.append(cur_cnt)

    '''
    print(len(cnt))
    for i in range(len(cnt)):
        print(len(cnt[i]))
    
    hulls = [cv.convexHull(c) for c in contours]
    print(hulls)
    cv.polylines(mask, hulls, True, (255, 0, 255), 2)
    cv.imshow('a',mask)
    cv.waitKey(0)

    '''
    hulls = [cv.convexHull(c) for c in contours]
    hull=[]
    for i in range(len(hulls)):
        cur_hull=np.zeros((len(hulls[i]),2))
        for j in range(len(hulls[i])):
            cur_hull[j]=hulls[i][j][0]
        hull.append(cur_hull)
    '''
    cv.polylines(mask, hulls, True, (0, 0, 255), 2)
    cv.imshow('a',mask)
    cv.waitKey(0)
    
    print(len(hull))
    for i in range(len(hull)):
        print(len(hull[i]))
    
    a=np.array(hull[0])
    print(a)
    '''
    return cnt, hull

def nonzero(depth):
    if np.sum(depth)<1:
        return [0]
    else:
        return depth[np.nonzero(depth)]

def create_glass_depth(path_depth,path_depth_glass,path_mask,k,write=True):
    
    depth_ori=cv.imread(path_depth,0)
    depth=np.zeros((depth_ori.shape[0]+k,depth_ori.shape[1]+k),'uint8')
    depth[k//2:depth_ori.shape[0]+k//2,k//2:depth_ori.shape[1]+k//2]=depth_ori

    glass_depth=np.zeros((depth_ori.shape[0],depth_ori.shape[1]),'uint8')

    _,hulls=glass_contour(path_mask)

    for h in range(len(hulls)):
        
        '''
        corner=np.array(([[86,414],     #00
                        [86,1548],     #01
                        [937,424],     #10
                        [937,1538]]))  #11
        '''
        
        hull=np.array(hulls[h])
        #uMin=np.min(hull[:,0])
        #uMax=np.max(hull[:,0])
        #vMin=np.min(hull[:,1])
        #vMax=np.max(hull[:,1])
        #print(hull)
        uMid=(np.min(hull[:,0])+np.max(hull[:,0]))/2
        vMid=(np.min(hull[:,1])+np.max(hull[:,1]))/2

        corner=np.array([[vMid,uMid],[vMid,uMid],[vMid,uMid],[vMid,uMid]],dtype=np.int32)
               
        #hull=np.array(sorted(hull,key=(lambda x:x[0])))
        #print(hull)

        for c in range(len(hull)): #hull[c][0] is u --- hull[c][1] is v
            #print(hull[c,0])
            #print(hull[c,1])
            #print(corner)
            length=np.sqrt(np.sum(np.square(hull[c]-(uMid,vMid))))
            lengths=np.array([
                np.sqrt(np.sum(np.square(corner[0]-(vMid,uMid)))),
                np.sqrt(np.sum(np.square(corner[1]-(vMid,uMid)))),
                np.sqrt(np.sum(np.square(corner[2]-(vMid,uMid)))),
                np.sqrt(np.sum(np.square(corner[3]-(vMid,uMid))))
            ])


            if hull[c,0]<uMid and hull[c,1]<vMid and length>lengths[0]: #00
                if corner[0,1]>hull[c,0]:
                    corner[0,1]=hull[c,0]
                if corner[0,0]>hull[c,1]:
                    corner[0,0]=hull[c,1]
            if hull[c,0]>uMid and hull[c,1]<vMid and length>lengths[1]: #01
                if corner[1,1]<hull[c,0]:
                    corner[1,1]=hull[c,0]
                if corner[1,0]>hull[c,1]:
                    corner[1,0]=hull[c,1]
            if hull[c,0]<uMid and hull[c,1]>vMid and length>lengths[2]: #10
                if corner[2,1]>hull[c,0]:
                    corner[2,1]=hull[c,0]
                if corner[2,0]<hull[c,1]:
                    corner[2,0]=hull[c,1]
            if hull[c,0]>uMid and hull[c,1]>vMid and length>lengths[3]: #11
                if corner[3,1]<hull[c,0]:
                    corner[3,1]=hull[c,0]
                if corner[3,0]<hull[c,1]:
                    corner[3,0]=hull[c,1]       

        corner_depth=np.array(
                        [np.median(nonzero(depth[corner[0,0]:corner[0,0]+k,corner[0,1]:corner[0,1]+k])),
                        np.median(nonzero(depth[corner[1,0]:corner[1,0]+k,corner[1,1]:corner[1,1]+k])),
                        np.median(nonzero(depth[corner[2,0]:corner[2,0]+k,corner[2,1]:corner[2,1]+k])),
                        np.median(nonzero(depth[corner[3,0]:corner[3,0]+k,corner[3,1]:corner[3,1]+k]))])
        '''
        corner_depth=np.array(
                        [np.median(depth[corner[0,0]:corner[0,0]+k,corner[0,1]:corner[0,1]+k]),
                        np.median(depth[corner[1,0]:corner[1,0]+k,corner[1,1]:corner[1,1]+k]),
                        np.median(depth[corner[2,0]:corner[2,0]+k,corner[2,1]:corner[2,1]+k]),
                        np.median(depth[corner[3,0]:corner[3,0]+k,corner[3,1]:corner[3,1]+k])])
        '''
        
        for v in range(depth_ori.shape[0]):
            for u in range(depth_ori.shape[1]):
                if v > (corner[0,0]-corner[1,0])/(corner[0,1]-corner[1,1])*(u-corner[0,1])+corner[0,0] \
                and v < (corner[2,0]-corner[3,0])/(corner[2,1]-corner[3,1])*(u-corner[2,1])+corner[2,0] \
                and u > (corner[0,1]-corner[2,1])/(corner[0,0]-corner[2,0])*(v-corner[0,0])+corner[0,1] \
                and u < (corner[1,1]-corner[3,1])/(corner[1,0]-corner[3,0])*(v-corner[1,0])+corner[1,1]:
                    dist=.0
                    cur=np.array([v,u])
                    for s in range(corner.shape[0]):
                        dist+=np.sqrt(np.sum(np.square(corner[s]-cur)))

                    for p in range(corner.shape[0]):
                        glass_depth[v,u]+=corner_depth[p]*(np.sqrt(np.sum(np.square(corner[p]-cur)))/dist)

    if write==True:
        cv.imwrite(path_depth_glass,glass_depth)
    
def create_glass_cloud(path_depth_glass,path_glass_cloud,write=True):

    glass=cv.imread(path_depth_glass,0)
    color=np.zeros((glass.shape[0],glass.shape[1],3),'uint8')
    color[:,:,:]=[[[128]*3]*glass.shape[1]]*glass.shape[0]

    glass=o3d.geometry.Image(glass)
    color=o3d.geometry.Image(color)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, glass, convert_rgb_to_intensity=False)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(1920, 1080, fx, fy, cx, cy)
    pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    #o3d.visualization.draw_geometries([pc])
    if write==True:
        o3d.io.write_point_cloud(path_glass_cloud, pc)

def merge_glass(path_cloud,path_glass_cloud,write=True):
    cloud=o3d.io.read_point_cloud(path_cloud)
    glass=o3d.io.read_point_cloud(path_glass_cloud)
    o3d.visualization.draw_geometries([cloud, glass])

#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------
#-----------------------------------------main-------------------------------------------
#----------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------


for i in range(6,18):
    print('img'+str(i))
    #starttime = datetime.datetime.now()

    name_left=path_img + 'rectified'+str(i)+'_left.jpg'
    name_right=path_img + 'rectified'+str(i)+'_right.jpg'
    name_depth=path_depth + 'rectified'+str(i)+'_depth.png'
    name_glass=path_glass + 'rectified'+str(i)+'_glass.png'
    name_mask=path_mask + 'rectified'+str(i)+'_mask.png'
    name_cloud=path_cloud + 'rectified'+str(i)+'_cloud.pcd'
    name_depth_masked=path_depth_masked + 'rectified'+str(i)+'_depth.png'
    name_cloud_glass=path_cloud_glass + 'rectified'+str(i)+'_cloud.pcd'
    name_depth_glass=path_depth_glass + 'rectified'+str(i)+'_depth.png'

    #create_depth(name_left,name_right,name_depth,True)
    #create_mask(name_glass,name_mask)
    #create_masked_depth(name_depth,name_mask,name_depth_masked)
    #create_cloud(name_depth_masked,name_left,name_cloud)
    #cloud_visualization(name_cloud)
    #create_glass_depth(name_depth,name_depth_glass,name_glass,7)
    create_glass_cloud(name_depth_glass,name_cloud_glass)
    merge_glass(name_cloud,name_cloud_glass)









    #endtime = datetime.datetime.now()
    #print(endtime - starttime).seconds

