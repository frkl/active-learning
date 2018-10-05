import os
import os.path
import sys

class Session:
	id=-1;
	def __init__(self,id=-1):
		if not os.path.isdir('sessions'):
			os.mkdir('sessions');
		if id==-1:
			#New session
			existing_sessions=dict();
			for dir in os.listdir('sessions'):
				existing_sessions[dir]=1;
			id=0;
			while True:
				if '%07d'%id in existing_sessions:
					id=id+1;
				else:
					break;
			os.mkdir(os.path.join('sessions','%07d'%id));
		elif not os.path.isdir(os.path.join('sessions','%07d'%id)):
			os.mkdir(os.path.join('sessions','%07d'%id));
		self.id=id;
		return;
	#
	def file(self,p1='',p2='',p3=''):
		if not os.path.isdir(os.path.join('sessions','%07d'%self.id)):
			os.mkdir(os.path.join('sessions','%07d'%self.id));
		if p1=='':
			return os.path.join('sessions','%07d'%self.id);
		elif p2=='':
			return os.path.join('sessions','%07d'%self.id,p1);
		elif p3=='':
			if not os.path.isdir(os.path.join('sessions','%07d'%self.id,p1)):
				os.mkdir(os.path.join('sessions','%07d'%self.id,p1));
			return os.path.join('sessions','%07d'%self.id,p1,p2);
		else:
			if not os.path.isdir(os.path.join('sessions','%07d'%self.id,p1)):
				os.mkdir(os.path.join('sessions','%07d'%self.id,p1));
			if not os.path.isdir(os.path.join('sessions','%07d'%self.id,p1,p2)):
				os.mkdir(os.path.join('sessions','%07d'%self.id,p1,p2));
			return os.path.join('sessions','%07d'%self.id,p1,p2,p3);
	
	def log(self,str,fname='log.txt'):
		print(str);
		sys.stdout.flush()
		f=open(self.file(fname),'a');
		f.write(str+'\n');
		f.close();
	
	def log_test(self,str):
		print(str);
		sys.stdout.flush()
		f=open(self.file('test.txt'),'a');
		f.write(str+'\n');
		f.close();
