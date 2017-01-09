
# coding: utf-8
# modified : 2016 / 10 / 30
# In[25]:
# in pic2numpy_TVT resize function has prob ! have to fix!!!!
# in savefic func name has to be change to more inspective ,  and savefic have prob 

from __future__ import unicode_literals
import pylab 
import matplotlib.pyplot as plt 
get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.image as mpimg
import random
import os 
from PIL import Image 
import png 
import dicom



# In[26]:

def crawl_folder(folder_path): #search_str 찾고자 하는 string 
    search_path_list=[]
    fileList = os.walk(folder_path).next()[2]
    subFolder_list = os.walk(folder_path).next()[1]
    if(len(fileList)!=0):
        for j in range(len(fileList)):
            search_path_list.append(folder_path+'/'+fileList[j])
    if len(subFolder_list)==0:
        return search_path_list
    else: 
        for i in range(len(subFolder_list)):
            search_path_list.extend(crawl_folder(folder_path+'/'+subFolder_list[i] ))
        return search_path_list


# In[27]:

def ConstraintSize(path_list , constraint_row , constraint_col ): #lst_ involves Path 
    print len(path_list)
    returend_path_list=[]
    count=0
    for ele in path_list:
        
        if type(ele)==str:
            if ".dcm" in ele:
                ds = dicom.read_file(ele)
                try:
                    pic_np=ds.pixel_array
                except ValueError as ve:
                    continue
            elif ".bmp" in ele:
                img=Image.open(ele)
                pic_np=np.array(img)
        
        elif  'numpy' in str(type(ele)):
            pic_np=ele

        else:
            img=Image.open(ele)
            pic_np=np.array(img)
        
        pic_row=np.shape(pic_np)[0]
   
        pic_col=np.shape(pic_np)[1]
     
        if pic_col < constraint_col  and  pic_row < constraint_row :
         
            returend_path_list.append(ele)
        count+=1
    print len(returend_path_list)
    return returend_path_list


# In[28]:

def pic2Numpy(source, save_path,batch_size, save_name ,img_size ,color_ch=1 , type_ = 'strict'):
    if type(source)==list:
        count=0
        batch_count=0
        n_lst=len(source)
        print 'list의 갯수 :' , n_lst
        train_img = np.zeros([batch_size , img_size[0] ,img_size[1] ,color_ch])
        print '총 파일의 갯수 : ' , len(source) 
        #print list_file
        print '만들어질 파일의 갯수 :' , int(len(source)/batch_size)
        print '버려지는 파일의 갯수 :' , int(len(source)%batch_size)
        for i in range(n_lst):
      
            ele=source[i]
            if (len(np.shape(ele))==2):
                ele=np.expand_dims(ele, 2) 
            train_img[count:count+1 , :,:,:] = ele
            count= count+1        
            if ( i!=0 and i%(batch_size-1)==0):
             
                np.save(save_path + '/'+save_name+'_'+str(batch_count) , train_img )
                batch_count = batch_count+1
                count=0


    else: #source type is folder 
        count =0
        batch_count =0
        global_count=0
        list_file=os.walk(source).next()[2]
        n_file = len(list_file)
        print '총 파일의 갯수 : ' , len(list_file) 
        #print list_file
        print '만들어질 파일의 갯수 :' , int(len(list_file)/batch_size)
        print '버려지는 파일의 갯수 :' , int(len(list_file)%batch_size)
        train_img = np.zeros([batch_size , img_size[0] ,img_size[1] ,color_ch])    
        for i in range(n_file):        
            img=Image.open(source+'/'+list_file[i])
            img=img.resize(img_size , Image.ANTIALIAS)
            img = np.array(img)

            if (len(np.shape(img))==2):
                img=np.expand_dims(img, 2) # add color dimension because grey img don't have any color ch 

            train_img[count:count+1 , :,:,:] = img
            count= count+1        
            if ( i!=0 and i%batch_size==0):
        
                np.save(save_path + '/'+save_name+'_'+str(batch_count) , train_img )
                batch_count = batch_count+1
                count=0


# In[29]:

def Set_ImageSize(source ):
    print "Set_ImageSize"
    if type(source)==list:
        n_lst = len(source)
        list_dcm=strInList(source , ".dcm")
        list_dcm.extend(strInList(source , ".bmp"))
  
        lst_dcm_array=[]
        returned_np_list=[]
        max_col=0;max_row=0;
        #list의 max_row 와 max_col을 구한다.
        for dcm_path in list_dcm:
            if ".dcm" in dcm_path: 
                ds=dicom.read_file(dcm_path) #open dicmom file
                try:           	
                    np_dcm=ds.pixel_array
                except ValueError as ve:
                    continue
            if ".bmp" in dcm_path:
                ds=Image.open(dcm_path)
                np_dcm = np.array(ds)
            lst_dcm_array.append(np_dcm)
            row, col=np.shape(np_dcm)
            #print row, col
            if row > max_row:
                max_row =row
            if col > max_col:
                max_col = col
        print 'max_row , max_col :' , max_row , max_col
        count=0
        for ele in lst_dcm_array:
            enlarged_np=np.zeros([max_row , max_col]) #drawing_paper 여기다가 이제 그릴거다.
            pic_row=np.shape(ele)[0]
            pic_col=np.shape(ele)[1]
            gap_row =max_row-pic_row
            gap_col =max_col-pic_col

            start_pt_col=int(gap_col/2)
            end_pt_col= int(gap_col/2)+pic_col
            start_pt_row=int(gap_row/2)
            end_pt_row= int(gap_row/2)+pic_row

            enlarged_np[start_pt_row : end_pt_row , :pic_col ] =  ele
            enlarged_np=np.roll(enlarged_np, start_pt_col ,axis=1)
            #plt.imshow(enlarged_np)
            
            returned_np_list.append(enlarged_np)
            
        return returned_np_list , max_row , max_col
        
    else:
            print "not yet build"
        


# In[ ]:




# In[30]:

def strInList(list_ , search_str):
    returnedList=[]
    for eleName in list_:
        if search_str in eleName:
            returnedList.append(eleName) 
    return returnedList


# In[31]:

def pic2numpy_TVT(path1 , path2 , train_rate , val_rate ,save_path ,img_row,img_col, \
                  color_ch,sort_type = 'random', aug=True , resize=True):
    
  
    n_class=2
    
    list_path_1=crawl_folder(path1)
    list_path_2=crawl_folder(path2)
    #dic type 변수를 만들어 거기에다가 path 와 
    dic_path_lab_all={}
    for ele in list_path_1:
        dic_path_lab_all[ele]=1
    for ele in list_path_2:
        dic_path_lab_all[ele]=2
    
    list_path_all=[]
    
    list_path_all.extend(list_path_1)
    list_path_all.extend(list_path_2)
    list_path_all=random.sample(list_path_all , len(list_path_all))
    
    
    n_list_path_all = len(list_path_all)
    n_train =int(train_rate*n_list_path_all) # number of train 
    n_val = int (val_rate*n_list_path_all) #number of val
    n_test= n_list_path_all-(n_train+n_val)
    print '모든 사진 갯수 ' , n_list_path_all
    print 'training  개수 : ' , n_train
    print 'validation  개수 : ',n_val
    print 'test  개수 : ',n_test
    
    train_count =0
    val_count=0
    test_count=0
    
    np_train=np.zeros([n_train , img_row , img_col , color_ch])
    np_train_lab = np.zeros([n_train , n_class])
    np_val=np.zeros([n_val , img_row , img_col , color_ch])
    np_val_lab = np.zeros([n_val , n_class])
    np_test=np.zeros([n_test , img_row , img_col , color_ch])
    np_test_lab = np.zeros([n_test , n_class])

    for ele in list_path_all:
    #define img_np 
        if ".dcm" in ele:
            img=dicom.read_file(ele)
            img_np=ds.pixel_array
        elif ".bmp" in ele: 
            img=Image.open(ele)
            if resize == True:
                img=img.resize((img_row , img_col) , Image.ANTIALIAS)
            img_np= np.array(img)
        elif ".jpg" in ele: 
            img=Image.open(ele)
            if resize == True:
                img=img.resize((img_row , img_col) , Image.ANTIALIAS)
            img_np= np.array(img)

        elif ".png" in ele:
            img=Image.open(ele)
            if resize == True:
                img=img.resize((img_row , img_col) , Image.ANTIALIAS)
            img_np= np.array(img)

        if len(np.shape(img_np))==2:
            img_np = np.expand_dims(img_np,2)
            
    #define lab_np 
    #img=mpimg.imread('stinkbug.png')
        if dic_path_lab_all[ele]==1:
            lab_np=0
        elif dic_path_lab_all[ele]==2:
            lab_np=1
        
        if train_count < n_train:
            np_train[train_count] =img_np
            np_train_lab[train_count , lab_np:lab_np+1] = 1
            train_count+=1

        elif val_count < n_val:
            np_val[val_count] = img_np
            np_val_lab[val_count , lab_np:lab_np+1]=1
            val_count +=1

        elif test_count < n_test:
            np_test[test_count] = img_np
            np_test_lab[test_count , lab_np:lab_np+1] = 1
            test_count +=1 
            
    np.save(save_path+'/'+'train_img',np_train)
    np.save(save_path+'/'+'train_lab',np_train_lab)
    np.save(save_path+'/'+'val_img',np_val)
    np.save(save_path+'/'+'val_lab',np_val_lab)
    np.save(save_path+'/'+'test_img',np_test)
    np.save(save_path+'/'+'test_lab',np_test_lab)
    
    




def pic2numpy_TVT_divide(divide_ind,path1 , path2 , train_rate , val_rate ,save_path ,img_row,img_col, color_ch,sort_type = 'random'):
    
    n_class=2
    
    list_path_1=crawl_folder(path1)
    list_path_2=crawl_folder(path2)
    #dic type 변수를 만들어 거기에다가 path 와 
    dic_path_lab_all={}
    for ele in list_path_1:
        dic_path_lab_all[ele]=1
    for ele in list_path_2:
        dic_path_lab_all[ele]=2
    
    list_path_all=[]
    
    list_path_all.extend(list_path_1)
    list_path_all.extend(list_path_2)
    list_path_all=random.sample(list_path_all , len(list_path_all))
    
    
    n_list_path_all = len(list_path_all)
    n_train =int(train_rate*n_list_path_all) # number of train 
    n_val = int (val_rate*n_list_path_all) #number of val
    n_test= n_list_path_all-(n_train+n_val)

    while(n_train%divide_ind!=0):
        n_train=n_train+1
        n_val=n_val-1
        
    print '모든 사진 갯수 ' , n_list_path_all
    print 'training  개수 : ' , n_train
    print 'validation  개수 : ',n_val
    print 'test  개수 : ',n_test
            
    
    
    
    
    train_count =0
    val_count=0
    test_count=0
    
    np_train=np.zeros([n_train/divide_ind , img_row , img_col , color_ch])
    np_train_lab = np.zeros([n_train/divide_ind , n_class])
    np_val=np.zeros([n_val , img_row , img_col , color_ch])
    np_val_lab = np.zeros([n_val , n_class])
    np_test=np.zeros([n_test , img_row , img_col , color_ch])
    np_test_lab = np.zeros([n_test , n_class])
    train_separation_ind=0
    print 'n_train/divide_ind' , n_train/divide_ind
    
    for ele in list_path_all:

    #define img_np 
        if ".dcm" in ele:
            img=dicom.read_file(ele)
            img_np=ds.pixel_array
        elif ".bmp" in ele: 
            img=Image.open(ele)
            img=img.resize((img_row , img_col) , Image.ANTIALIAS)
            img_np= np.array(img)
        elif ".png" in ele:
            img=Image.open(ele)
            img=img.resize((img_row , img_col) , Image.ANTIALIAS)
            img_np= np.array(img)
            
        if len(np.shape(img_np))==2:
            img_np = np.expand_dims(img_np,2)
            
    #define lab_np 
    #img=mpimg.imread('stinkbug.png')
        if dic_path_lab_all[ele]==1:
            lab_np=0
        elif dic_path_lab_all[ele]==2:
            lab_np=1
        print train_count
        if train_count < (n_train/divide_ind) :
            np_train[train_count] =img_np
            np_train_lab[train_count] = lab_np
            train_count+=1
            if train_count==(n_train/divide_ind) and train_separation_ind < divide_ind: 
                np.save(save_path+'/'+'train_'+str(train_separation_ind)+'_img',np_train)
                np.save(save_path+'/'+'train_'+str(train_separation_ind)+'_lab',np_train_lab)
                np_train=np.zeros([n_train/divide_ind , img_row , img_col , color_ch])
                np_train_lab = np.zeros([n_train/divide_ind , n_class])
                train_separation_ind +=1            
                if train_separation_ind == divide_ind:
                    continue
                train_count=0
                
            
        elif val_count < n_val:
            print 'val_count'
            np_val[val_count] = img_np
            np_val_lab[val_count]=lab_np
            val_count +=1

        elif test_count < n_test:
            print 'test_count'
            np_test[test_count] = img_np
            np_test_lab[test_count] = lab_np
            test_count +=1 
            
    np.save(save_path+'/'+'val_img',np_val)
    np.save(save_path+'/'+'val_lab',np_val_lab)
    np.save(save_path+'/'+'test_img',np_test)
    np.save(save_path+'/'+'test_lab',np_test_lab)
    
def dcm2scaled_np(dcm_path , scale=True):
    print ("ordr : color , row , col ")
    plan = dicom.read_file(dcm_path)
    shape = plan.pixel_array.shape
    dcm_np = plan.pixel_array
    img_2d = []
    
    if len(shape)==3: # color ch        
        
        shape=np.shape(dcm_np)
    	if scale != True:
		return dcm_np    
        #check max_value
        for i in range(len(dcm_np)):
            mat=dcm_np[i]
            
            print 'np.shape(mat)',np.shape(mat)
            max_val=0
            for row in mat:
                for col in row:
                    if col > max_val:
                        max_val = col
                        print 'max value was changed ,Max Value:',max_val
            scaled_mat=((mat.astype(float))/float(max_val)*255)
            dcm_np[i]=scaled_mat.astype(int)
        return dcm_np
             
    elif len(shape)==2:
        print "not yet bulid!"

def dcm2pic(dcm_np , save_path , save_name , extension , img_row , img_col):
    #color_ch chanege 
    color_ch , row , col =np.shape(dcm_np)
    dcm_np=dcm_np.reshape(row , col ,color_ch)
    img=Image.fromarray(dcm_np)
    img_size=(img_row , img_col)
    img=img.resize(img_size , Image.ANTIALIAS)
    plt.imshow(img)
    plt.savefig(save_path + save_name+extension)
    print 'save clear'
    
   
   
def savepic(save_path,extension,img_source,lab_source, pred_list , result_list, img_row , img_col,color_ch=1 ):
    assert 'numpy' in str(type(img_source)), "numpy type has to be numpy or list "
    
    
    
    
    print "we are in savepic"
    
    img_list=[]
    lab_list=[]
    count=0
    if type(lab_source )== str:
        lab_np=np.load(lab_source)
        if len(np.shape(lab_np))==2: # lab_np shape has to be (None , 2)
            for count in range(len(lab_source)):
                lab_list.append(np.argmax(lab_source[count: count+1]))
    elif type(lab_source) == list:
        lab_list=lab_source
        
        
    count=0
    if type(img_source)==str: ## if you load .npy file 
        img_np=np.load(img_source)
    elif 'numpy' in str(type(img_source)):
        img_np = img_source
       
    if len(np.shape(img_np))==4: ## load numpy shape has to be (None , img_row , img_col , color_ch )
        for count in range(len(img_source)):
            img_list.append(img_source[count : count+1].reshape(img_row, img_col , color_ch))        
          	
            
    ##making result_list###     
    if result_list == None:
        lab_pred_zip = zip(lab_list , pred_list)
        for lab_ele,pred_ele in lab_pred_zip:
            if lab_ele == pred_ele: result_list.append('True')
            elif lab_ele != pred_ele: result_list.append('False')
    
    #########################
    all_list_zip = zip(img_list ,lab_list ,pred_list ,result_list)
    
    
    ###making save file to folder define with save path###
    index=0
    
    for img_ele ,lab_ele , pred_ele , result_ele in  all_list_zip :
	try:
		img_ele=img_ele.reshape(img_row , img_col) 
		file_str =str(index)+" lab : "+str(lab_ele)+":pred : "+str(pred_ele)+"result : "+str(result_ele)+extension
		print file_str
		plt.imshow(img_ele,cmap='Greys_r')
		plt.savefig(save_path+"/"+file_str )
		index+=1
	except:
		continue
	    #source_list 는 
	    # num x pred : 1 label x
	    
		        
                

    








