import numpy as np
import math as m

FACE_B=0; FACE_D=1; FACE_F=2; FACE_L=3; FACE_R=4; FACE_T=5

def rotx(ang):
	return np.array([[1,0,0],[0,np.cos(ang),-np.sin(ang)],[0,np.sin(ang),np.cos(ang)]])

def roty(ang):
	return np.array([[np.cos(ang),0,np.sin(ang)],[0,1,0],[-np.sin(ang),0,np.cos(ang)]])

def rotz(ang):
	return np.array([[np.cos(ang),-np.sin(ang),0],[np.sin(ang),np.cos(ang),0],[0,0,1]])

def xy2angle(XX,YY,im_w,im_h):
	_XX=2*(XX+0.5)/float(im_w)-1
	_YY=1-2*(YY+0.5)/float(im_h)
	theta=_XX*np.pi
	phi=_YY*(np.pi/2)
	return theta,phi

def to_3dsphere(theta,phi,R):
	x=R*np.cos(phi)*np.cos(theta)
	y=R*np.sin(phi)
	z=R*np.cos(phi)*np.sin(theta)
	return x,y,z

def pruned_inf(angle):
	eps=1e-8
	angle[angle==0.0]=eps
	angle[angle== np.pi]= np.pi-eps
	angle[angle==-np.pi]=-np.pi+eps
	angle[angle== np.pi/2]= np.pi/2-eps
	angle[angle==-np.pi/2]=-np.pi/2+eps
	return angle

def get_face(x,y,z,face_map):
	eps=1e-8
	max_arr=np.maximum.reduce([np.abs(x),np.abs(y),np.abs(z)])
	x_faces=(max_arr-np.abs(x))<eps
	y_faces=(max_arr-np.abs(y))<eps
	z_faces=(max_arr-np.abs(z))<eps
	face_map[(x>=0)&x_faces]=FACE_F
	face_map[(x<=0)&x_faces]=FACE_B
	face_map[(y>=0)&y_faces]=FACE_T
	face_map[(y<=0)&y_faces]=FACE_D
	face_map[(z>=0)&z_faces]=FACE_R
	face_map[(z<=0)&z_faces]=FACE_L
	return face_map

def face_to_cube_coord(face_gr,x,y,z):
	h,w=face_gr.shape
	direct=np.zeros((h,w,3))
	fb=(face_gr==FACE_F); direct[fb,0]= z[fb]; direct[fb,1]= y[fb]; direct[fb,2]= x[fb]
	bb=(face_gr==FACE_B); direct[bb,0]=-z[bb]; direct[bb,1]= y[bb]; direct[bb,2]= x[bb]
	tb=(face_gr==FACE_T); direct[tb,0]= z[tb]; direct[tb,1]=-x[tb]; direct[tb,2]= y[tb]
	db=(face_gr==FACE_D); direct[db,0]= z[db]; direct[db,1]= x[db]; direct[db,2]= y[db]
	rb=(face_gr==FACE_R); direct[rb,0]=-x[rb]; direct[rb,1]= y[rb]; direct[rb,2]= z[rb]
	lb=(face_gr==FACE_L); direct[lb,0]= x[lb]; direct[lb,1]= y[lb]; direct[lb,2]= z[lb]
	x_oncube=(direct[:,:,0]/np.abs(direct[:,:,2])+1)/2
	y_oncube=(-direct[:,:,1]/np.abs(direct[:,:,2])+1)/2
	return x_oncube,y_oncube

def norm_to_cube(out_coord,w):
	out=out_coord*(w-1)
	np.clip(out,0.0,w-1,out=out)
	return out
