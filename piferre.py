#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Interface to use FERRE from python for DESI/BOSS data

use: piferre -sp path-to-spectra [-rv rvpath -l libpath -spt sptype -rvt rvtype -c config -n cores  -t truthfile -o list_of_targets -x]

e.g. piferre -sp /data/spectro/redux/dc17a2/spectra-64 

Author C. Allende Prieto
'''
import pdb
import sys
import os
import platform
import glob
import re
import importlib
from numpy import arange, loadtxt, savetxt, genfromtxt, zeros, ones, nan, sqrt, interp,     \
  concatenate, correlate, array, reshape, min, max, diff, where, divide, mean, stack, vstack, \
  int64, int32,  \
  log10, median, std, mean, pi, intersect1d, isfinite, ndim, cos, sin, exp, isnan
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.table as tbl
import astropy.units as units
import matplotlib.pyplot as plt
from synple import head_synth,lambda_synth
import subprocess
import datetime, time
import argparse
import yaml
from multiprocessing import Pool,cpu_count

version = '0.4'
hplanck = 6.62607015e-34 # J s
clight = 299792458.0 #/s
piferredir = os.path.dirname(os.path.realpath(__file__))
confdir = os.path.join(piferredir,'config')
filterdir = os.path.join(piferredir,'filter')


#create a slurm script for a given pixel
def write_slurm(root,ncores=1,minutes=102,path=None,ngrids=None, 
config='desi-n.yaml', cleanup=True):

    ferre=os.environ['HOME']+"/ferre/src/a.out"
    python_path=os.environ['HOME']+"/piferre"
    try: 
      host=os.environ['HOST']
    except:
      host='Unknown'

    conf=load_conf(config,confdir=confdir)

    now=time.strftime("%c")
    if path is None: path='.'
    if ngrids is None: ngrids=1

    f=open(os.path.join(path,root+'.slurm'),'w')
    f.write("#!/bin/bash \n")
    f.write("#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# \n")
    f.write("#This script was written by piferre.py version "+version+" on "+now+" \n") 
    f.write("#SBATCH --time="+str(int(minutes)+1)+"\n") #minutes
    f.write("#SBATCH --ntasks=1" + "\n")
    f.write("#SBATCH --nodes=1" + "\n")
    if host[:4] == 'cori': #cori
      f.write("#SBATCH --qos=regular" + "\n")
      f.write("#SBATCH --constraint=haswell" + "\n")
      #f.write("#SBATCH --time="+str(minutes)+"\n") #minutes
      #f.write("#SBATCH --ntasks=1" + "\n")
      f.write("#SBATCH --cpus-per-task="+str(ncores*2)+"\n")
      f.write("#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# \n")
    elif (host == 'login1'): #lapalma
      #f.write("#SBATCH  -J "+str(root)+" \n")
      #f.write("#SBATCH  -o "+str(root)+"_%j.out"+" \n")
      #f.write("#SBATCH  -e "+str(root)+"_%j.err"+" \n")
      f.write("#SBATCH --cpus-per-task="+str(ncores)+"\n")
      #hours2 = int(minutes/60) 
      #minutes2 = minutes%60
      #f.write("#SBATCH  -t "+"{:02d}".format(hours2)+":"+"{:02d}".format(minutes2)+":00"+"\n") #hh:mm:ss
      #f.write("#SBATCH  -D "+os.path.abspath(path)+" \n")    
      f.write("#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# \n")
      f.write("module load gnu"+"\n")
      f.write("module load python/3.8"+"\n")
    else: # perlmutter
      f.write("#SBATCH --qos=regular" + "\n")
      f.write("#SBATCH --constraint=cpu" + "\n")
      #f.write("#SBATCH --time="+str(minutes)+"\n") #minutes
      #f.write("#SBATCH --ntasks=1" + "\n")
      f.write("#SBATCH --account=desi \n")
      f.write("#SBATCH --cpus-per-task="+str(ncores*2)+"\n")
      f.write("#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-# \n")
      f.write("module load PrgEnv-gnu"+"\n")
      f.write("module load python"+"\n")


    f.write("cd "+os.path.abspath(path)+"\n")
    f.write("vmstat 1 > "+str(root)+"_vmstat.dat & \n")
    f.write("vmstat_pid=$! \n")
    for i in range(ngrids):
      #f.write("cp input.nml-"+root+"_"+str(i)+" input.nml \n")
      f.write("(  time "+ferre+" -l input.lst-"+root+"_"+"{:02d}".format(i+1)+" >& "+root+".log_"+"{:02d}".format(i+1))
      f.write(" ; echo FERRE job " + "{:02d}".format(i+1) + " ) & \n")
      f.write("pids+=($!) \n")
    f.write("for pid in ${pids[@]} \n")
    f.write("do \n")
    f.write("  wait $pid \n")
    f.write("done \n")
    f.write("kill $vmstat_pid \n")
    command="python3 -c \"import sys; sys.path.insert(0, '"+python_path+ \
     "'); from piferre import opfmerge, oafmerge, write_tab_fits, write_mod_fits, cleanup;"+ \
     " opfmerge(\'"+str(root)+"\',config='"+config+"\');"
    if 'elem' in conf:
      command=command + " oafmerge(\'"+str(root)+"\',config='"+config+"\');"
    command=command + " write_tab_fits(\'"+str(root)+"\',config='"+config+"\');"+ \
     " write_mod_fits(\'"+str(root)+"\',config='"+config+"\')"
    if cleanup == True: 
      command=command+" ; cleanup(\'"+str(root)+"\')"
    command=command+" \"\n"
    f.write(command)
    f.close()
    os.chmod(os.path.join(path,root+'.slurm'),0o755)

    return None


#identify all the *.slurm files in the path and merge them in groups of nmerge so that there are fewer/longer jobs. The scripts are named job-*.slurm and written to the current folder
def mergeslurm(path='./',ext='slurm',nmerge=2,concurrent=False):

  slurms = glob.glob(os.path.join(path,'**','*'+ext), recursive=True)

  nfiles = len(slurms)

  k = 0 
  wtime = -1
  for i in range(nfiles):
    f1 = open(slurms[i],'r')
    j = i % nmerge
    if j == 0:
      k = k + 1
      if k > 1: 
        if wtime > -1:
          entries = header[wtime].split('=')
          header[wtime] = entries[0]+'='+str(time)+'\n'
        f2.writelines(header)
        if concurrent: body.append("wait\n")
        f2.writelines(body)
        f2.close()
      f2 = open('job-'+"{:04d}".format(k)+'.slurm','w')
      time = 0
      header = []
      body = []
    if concurrent: 
      body.append("(\n")
    for line in f1: 
      if line[0] == "#":
        if j == 0: header.append(line)
        if '--time' in line:
          entries = line.split('=') 
          time = time + int(entries[1])
          if j == 0: wtime = len(header)-1
      else:
        body.append(line)
    if concurrent:
      body.append(") & \n")
  
  if wtime > -1: 
    entries = header[wtime].split('=')
    header[wtime] = entries[0]+'='+str(time)+'\n' 
  f2.writelines(header)
  if concurrent: body.append("wait\n")
  f2.writelines(body)
  f2.close()
    

  print(slurms)

  return None


#remove FERRE I/O files after the final FITS tables have been produced
def cleanup(root):
  vrdfiles = glob.glob(root+'*vrd')
  wavefiles = glob.glob(root+'*wav')
  opffiles = glob.glob(root+'*opf*')
  optfiles = glob.glob(root+'*opt*')
  nmlfiles = glob.glob('input.nml-'+root+'*')
  lstfiles = glob.glob('input.lst-'+root+'*')
  errfiles = glob.glob(root+'*err')
  frdfiles = glob.glob(root+'*frd')
  nrdfiles = glob.glob(root+'*nrd*')
  mdlfiles = glob.glob(root+'*mdl*')
  ndlfiles = glob.glob(root+'*ndl*')
  fmpfiles = glob.glob(root+'*fmp.fits')
  scrfiles = glob.glob(root+'*scr.fits')
  logfiles = glob.glob(root+'.log*')
  slurmfiles = glob.glob(root+'*slurm')
  oaffiles = glob.glob(root+'*.oaf.*')
  nadfiles = glob.glob(root+'*.nad.*')
  nalfiles = glob.glob(root+'*.nal.*')
  allfiles = vrdfiles + wavefiles + opffiles + optfiles + nmlfiles + lstfiles + \
              errfiles + frdfiles + nrdfiles + mdlfiles + ndlfiles + \
              fmpfiles + scrfiles + logfiles + slurmfiles + oaffiles + nadfiles + nalfiles 

  print('removing files:',end=' ')
  for entry in allfiles: 
    print(entry,' ')
    os.remove(entry)
  print(' ')

  return
  
#create a FERRE control hash (content for a ferre input.nml file) and write it to disk
def mknml(conf,root,libpath='.',path='.'):

  try: 
    host=os.environ['HOST']
  except:
    host='Unknown'

  try: 
    scratch=os.environ['SCRATCH']
  except:
    scratch='./'


  bands=conf['bands']
  grids=conf['grids']
  if 'grid_bands' in conf: grid_bands=conf['grid_bands']
  if 'abund_grids' in conf: abund_grids=conf['abund_grids']

  for k in range(len(grids)): #loop over all grids
    synth=grids[k]
    synthfiles=[]
    if 'grid_bands' in conf:
      grid_bands=conf['grid_bands']	
      for band in grid_bands:
        if len(grid_bands) == 0:
          gridfile=synth+'.dat'
        else:
          if len(grid_bands) == 1:
            gridfile=synth+'-'+band+'.dat'
          elif len(grid_bands) == len(bands):
            gridfile=synth+'-'+band+'.dat'
          else:
            print('mknml: error -- the array grid_bands must have 0, 1 or the same length as bands')
            return None       
        synthfiles.append(gridfile)
    else:
      gridfile=synth+'.dat'  
      synthfiles.append(gridfile)

    libpath=os.path.abspath(libpath)
    header=head_synth(os.path.join(libpath,synthfiles[0]))
    if ndim(header) == 0:
      nd=int(header['N_OF_DIM'])
      n_p=tuple(array(header['N_P'].split(),dtype=int))
    else:
      nd=int(header[0]['N_OF_DIM'])
      n_p=tuple(array(header[0]['N_P'].split(),dtype=int))


    lst=open(os.path.join(path,'input.lst-'+root+'_'+"{:02d}".format(k+1)),'w')
    for run in conf[synth]: #loop over all runs (param and any other one)

      #global keywords in yaml adopted first
      nml=dict(conf['global'])

      #adding/override with param keywords in yaml
      if 'param' in conf[synth]:
        for key in conf[synth]['param'].keys(): nml[key]=conf[synth]['param'][key]

      #adding/override with run keywords in yaml 
      for key in conf[synth][run].keys(): 
        nml[key] = str(conf[synth][run][key])

      #check that inter is feasible with this particular grid
      if 'inter' in nml:
        if (min(n_p)-1 < int(nml['inter'])): nml['inter']=min(n_p)-1
      #ncores from command line is for the slurm job, 
      #the ferre omp nthreads should come from the config yaml file
      nml['ndim']=nd
      for i in range(len(synthfiles)): 
        nml['SYNTHFILE('+str(i+1)+')'] = "'"+os.path.join(libpath,synthfiles[i])+"'"

      #extensions provided in yaml for input/output files are now supplemented with root
      files = ['pfile', 'ffile', 'erfile','opfile','offile','sffile']
      for entry in files:
        if entry in nml: nml[entry] = "'"+root+'.'+nml[entry]+"'"
      if 'filterfile' in nml: nml['filterfile'] = "'"+os.path.join(filterdir,nml['filterfile'])+"'"


      #make sure tmp 'sort' files are stored in $SCRATCH for cori
      #if host[:4] == 'cori':
      #  nml['scratch'] = "'"+scratch+"'"
      #no longer needed after replacing fsort by msort in ferre (may 2022)

      #get rid of keywords in yaml that are not for the nml file, but for opfmerge 
      #or write_tab
      labels = nml['labels']
      if 'labels' in nml: del nml['labels']
      if 'llimits' in nml: del nml['llimits']
      if 'steps' in nml: del nml['steps']

      #expanding abbreviations $i -> synth number, $synth -> grid name
      # $Teff -> index of Teff variable, etc.
      for key in nml:
        nml[key] = nml_key_expansion(str(nml[key]),k,synth,labels)


      nmlfile='input.nml-'+root+'_'+"{:02d}".format(k+1)+run
      lst.write(nmlfile+'\n')
      write_nml(nml,nmlfile=nmlfile,path=path)

    if 'extensions' in conf:
      for run in conf['extensions']: #loop over all extensions 
        #extensions are like runs to be applied to all grids

        #global keywords in yaml adopted first
        nml=dict(conf['global'])

        #adding/override with param keywords in yaml
        if 'param' in conf[synth]:
          for key in conf[synth]['param'].keys(): nml[key]=conf[synth]['param'][key]

        #adding/override with run keywords in yaml 
        for key in conf['extensions'][run].keys(): 
          nml[key] = str(conf['extensions'][run][key])

        #check that inter is feasible with this particular grid
        if 'inter' in nml:
          if (min(n_p)-1 < int(nml['inter'])): nml['inter']=min(n_p)-1
        #ncores from command line is for the slurm job, 
        #the ferre omp nthreads should come from the config yaml file
        nml['ndim']=nd
        for i in range(len(synthfiles)): 
          nml['SYNTHFILE('+str(i+1)+')'] = "'"+os.path.join(libpath,synthfiles[i])+"'"

        #extensions provided in yaml for input/output files are now supplemented with root
        files = ['pfile', 'ffile', 'erfile','opfile','offile','sffile']
        for entry in files:
          if entry in nml: nml[entry] = "'"+root+'.'+nml[entry]+"'"
        if 'filterfile' in nml: nml['filterfile'] = "'"+os.path.join(filterdir,nml['filterfile'])+"'"


        #make sure tmp 'sort' files are stored in $SCRATCH for cori
        #no longer needed after changing fsort by msort
        #if host[:4] == 'cori':
        #  nml['scratch'] = "'"+scratch+"'"

        #get rid of keywords in yaml that are not for the nml file, but for opfmerge or write_tab
        labels = nml['labels']
        if 'labels' in nml: del nml['labels']
        if 'llimits' in nml: del nml['llimits']
        if 'steps' in nml: del nml['steps']

        #expanding abbreviations $i -> synth number, $synth -> grid name
        # $Teff -> index of Teff variable, etc.
        for key in nml:
          nml[key] = nml_key_expansion(str(nml[key]),k,synth,labels)

        if run == "abund":
          if synth not in abund_grids: continue #skip abundances for grids not in abund_grids

          #replacing $elem -> element symbol
          #          $proxy -> index of the proxy variable for the element in the grid
          proxies=conf['proxy']
          indproxies = zeros(len(proxies),dtype=int)
          j = 0
          for entry in proxies:
            i = 1
            for entry2 in labels:
              if entry == entry2: indproxies[j] = i
              i = i + 1
            j = j + 1

          for i in range(len(conf['elem'])):
            nml1 = dict(nml)
            for key in nml1: 
              content = str(nml[key])
              if '$elem' in content: content = content.replace('$elem',str(conf['elem'][i]))
              if '$proxy' in content: content = content.replace('$proxy',str(indproxies[i]))
              nml1[key] = content

            nmlfile='input.nml-'+root+'_'+"{:02d}".format(k+1)+conf['elem'][i]
            lst.write(nmlfile+'\n')
            write_nml(nml1,nmlfile=nmlfile,path=path)
        else:
          nmlfile='input.nml-'+root+'_'+"{:02d}".format(k+1)+run
          lst.write(nmlfile+'\n')
          write_nml(nml,nmlfile=nmlfile,path=path)

    lst.close()

  return None

def nml_key_expansion(content,k,synth,labels):
    if '$i' in content: content = content.replace('$i',"{:02d}".format(k+1))
    if '$synth' in content: content = content.replace('$synth',synth)
    i = 1
    for entry in labels:
      ss = '$'+entry
      if ss in content: content = content.replace(ss,str(i))
      i = i + 1

    return content

#write out a FERRE control hash to an input.nml file
def write_nml(nml,nmlfile='input.nml',path=None):
    if path is None: path='./'
    f=open(os.path.join(path,nmlfile),'w')
    f.write('&LISTA\n')
    for item in nml.keys():
        f.write(str(item))
        f.write("=")
        f.write(str(nml[item]))
        f.write("\n")
    f.write(" /\n")
    f.close()
    return None

#run FERRE
def ferrerun(path=None):
    if path is None: path="./"
    pwd=os.path.abspath(os.curdir)
    os.chdir(path)
    ferre="/home/callende/ferre/src/a.out"
    code = subprocess.call(ferre)
    os.chdir(pwd)
    return code


#read redshift derived by the DESI pipeline
def read_zbest(filename):
  hdu=fits.open(filename)
  if len(hdu) > 1:
    enames=extnames(hdu)
    #print(enames)
    if 'ZBEST' in enames:
      zbest=hdu['zbest'].data
      zerr=zbest['zerr']
      targetid=zbest['targetid'] #array of long integers
    elif 'REDSHIFTS' in enames:
      zbest=hdu['redshifts'].data
      zerr=zbest['zerr']
      targetid=zbest['targetid'] #array of long integers
    else:
      zbest=hdu[1].data
      zerr=zbest['z_err']
      plate=zbest['plate']
      mjd=zbest['mjd']
      fiberid=zbest['fiberid']
      targetid=[]
      for i in range(len(plate)): 
        targetid.append(str(plate[i])+'-'+str(mjd[i])+'-'+str(fiberid[i]))
      targetid=array(targetid)  #array of strings

    #print(type(targetid),type(zbest['z']),type(targetid[0]))
    #print(targetid.shape,zbest['z'].shape)
    z=dict(zip(targetid, zip(zbest['z'],zerr) ))
  else:
    z=dict()

  return(z)

#read redshift derived by the Koposov pipeline
def read_k(filename):
  hdu=fits.open(filename)
  if len(hdu) > 1:
    k=hdu['rvtab'].data
    targetid=k['targetid']
    zerr=k['vrad_err']/clight*1e3
    #targetid=k['fiber']
    #teff=k['teff']
    #logg=k['loog']
    #vsini=k['vsini']
    #feh=k['feh']
    #z=k['vrad']/clight*1e3
    z=dict(zip(targetid, zip(k['vrad']/clight*1e3, zerr) ))  
    #z_err=dict(zip(k['target_id'], k['vrad_err']/clight*1e3))  
  else:
    z=dict() 
  return(z)

#read truth tables (for simulations)
def read_truth(filename):
  hdu=fits.open(filename)
  truth=hdu[1].data
  targetid=truth['targetid']
  feh=dict(zip(targetid, truth['feh']))
  teff=dict(zip(targetid, truth['teff']))
  logg=dict(zip(targetid, truth['logg']))
  #rmag=dict(zip(targetid, truth['flux_r']))
  rmag=dict(zip(targetid, truth['mag']))
  z=dict(zip(targetid, truth['truez']))
  return(feh,teff,logg,rmag,z)

#read spectra
def read_spec(filename,band=None):

  hdu=fits.open(filename)

  if filename.find('spectra-') > -1 or filename.find('exp_') > -1 or filename.find('coadd') > -1: #DESI
    wavelength=hdu[band+'_WAVELENGTH'].data #wavelength array
    flux=hdu[band+'_FLUX'].data       #flux array (multiple spectra)
    ivar=hdu[band+'_IVAR'].data       #inverse variance (multiple spectra)
    #mask=hdu[band+'_MASK'].data       #mask (multiple spectra)
    res=hdu[band+'_RESOLUTION'].data  #resolution matrix (multiple spectra)
    #bintable=hdu['BINTABLE'].data  #bintable with info (incl. mag, ra_obs, dec_obs)

  if filename.find('spPlate') > -1: #SDSS/BOSS
    header=hdu['PRIMARY'].header
    wavelength=header['CRVAL1']+arange(header['NAXIS1'])*header['CD1_1'] 
#wavelength array
    wavelength=10.**wavelength
    flux=hdu['PRIMARY'].data       #flux array (multiple spectra)
    #ivar=hdu['IVAR'].data       #inverse variance (multiple spectra)    
    ivar=hdu[1].data       #inverse variance (multiple spectra)
    #andmask=hdu['ANDMASK'].data       #AND mask (multiple spectra)  
    #ormask=hdu['ORMASK'].data       #OR mask (multiple spectra)
    #res=hdu['WAVEDISP'].data  #FWHM array (multiple spectra)
    res=hdu[4].data  #FWHM array (multiple spectra)
    #bintable=hdu['BINTABLE'].data  #bintable with info (incl. mag, ra, dec)
    

  return((wavelength,flux,ivar,res))

#read sptab/rvtab file, returning sp/rvtab, fibermap and primary header (s,f,p)
def read_tab(file):

  d=fits.open(file)
  n=extnames(d)
  h=d[0].header
  if 'SPTAB' in n or 'sptab' in n:
    s=d['sptab'].data
  elif 'rvtab' in n or 'RVTAB' in n:
    s=d['rvtab'].data
  else:
    print('Error: cannot find an rvtab or sptab extension in the file')
    s=None
  f=d['fibermap'].data
 
  return(s,f,h) 

#read sp/rvmod table, returning bx,by,rx,ry,zx,zy (wavelength and fluxes), and 
#primary header (h)
#spmod files contain obs,err,flx,fit and abu arrays
#rvmod files contain only fit 
def read_mod(file):

  d=fits.open(file)
  h=d[0].data
  bx=d['b_wavelength'].data
  by=d['b_model'].data
  rx=d['r_wavelength'].data
  ry=d['r_model'].data
  zx=d['z_wavelength'].data
  zy=d['z_model'].data

  return(bx,by,rx,ry,zx,zy,h)
 

#show the data (obs) and best-fitting model (fit) for i-th spectrum (0,1...) in an rv/spmod file
def show1(modfile,i=0,abu=False):
  
  xb,yb,xr,yr,xz,yz,h = read_spmod(modfile)
  plt.clf()
  plt.ion()
  plt.plot(xb,yb['obs'][i,:],xr,yr['obs'][i,:],xz,yz['obs'][i,:])
  plt.plot(xb,yb['fit'][i,:],xr,yr['fit'][i,:],xz,yz['fit'][i,:])
  plt.xlabel('wavelength ($\AA$)')
  plt.ylabel('flux')
  if abu == True:
    plt.plot(xb,yb['abu'][i,:],xr,yr['abu'][i,:],xz,yz['abu'][i,:])
    #plt.plot(xb,yb['abu'][i,:]/yb['fit'][i,:],xr,yr['abu'][i,:]/yr['fit'][i,:],xz,yz['abu'][i,:]/yz['fit'][i,:])
    plt.legend(['b','r','z','fit b','fit r','fit z','abu b','abu r','abu z'])
  else:
    plt.legend(['b','r','z','fit b','fit r','fit z'])
  plt.show()

  return(None)

def smooth(x,n):

  """Smooth using a Svitzky-Golay cubic filter


  Parameters
  ----------
  x: arr
    input array to smooth
  n: int
    window size
  """

  x2 = savgol_filter(x, n, 3)

  return(x2)


#identify suitable calibration stars in an sframe by matching to an sptab, 
#returning their indices in the sframe and sptab arrays
def ind_calibrators(sframe,sptab,
                    tefflim=[3600.,10000.],gaiaglim=[15.,20.],maxchi=1e5):
  """
  sframe = sframe file
  sptab =  sptabfile
  """

  #read fibermap from sframefile

  sf=fits.open(sframe)
  fmp=sf['fibermap'].data

  #info on calibration stars from sptab
  s,f,h = read_tab(sptab)
  ind = {}  #dict that connects targetid to index of spectrum in spmod/tab
  for i in range(len(f['target_ra'])): ind[f['targetid'][i]] = i

  i = 0
  j = 0
  ind_sf = []
  ind_sp = []
  for entry in fmp['targetid']:
    if entry in ind.keys():
      ie = ind[entry]
      if (s['teff'][ie] > tefflim[0] and s['teff'][ie] <= tefflim[1] and
        fmp['gaia_phot_g_mean_mag'][j] > gaiaglim[0] and
        fmp['gaia_phot_g_mean_mag'][j] < gaiaglim[1] and
        s['chisq_tot'][ind[entry]] < maxchi):
          ind_sf.append(j)
          ind_sp.append(ie)
          i += 1
    j += 1

  return(ind_sf,ind_sp)

#calibrate in flux an sframe using parameters in sptab and model flux in spmod
def reponse(ind_sf,ind_sp,sframe,spmod):
  """
  ind_sf = indices for calibrators in sframe
  ind_sp = indices for calibrators in sptab/spmod
  sframe = sframe file
  spmod = piferre spmod produced with theoretical SEDs in the 'fit' field

  """

  from extinction import ccm89, apply
  from pyphot import get_library

  lib = get_library()
  filter = lib['Gaia_MAW_G']

  #info on calibration stars
  bx,by,rx,ry,zx,zy,h2 = read_spmod(spmod)

  #observations
  sf=fits.open(sframe)
  fmp=sf['fibermap'].data
  x=sf['wavelength'].data
  y=sf['flux'].data
  ivar=sf['ivar'].data

  #ext = F99(Rv=3.1)

  k = 0
  for i in ind_sp: 
      j = ind_sf[k]
      if x[0] < 4000.: 
        model = interp(x,bx,by['fit'][i,:])
      elif x[0] < 6000.: 
        model = interp(x,rx,ry['fit'][i,:])
      else: 
        model = interp(x,zx,zy['fit'][i,:])

      newx = x.byteswap().newbyteorder() # force native byteorder for calling ccm89
      model = model * 4. * pi  # Hlambda to Flambda
      model = apply( ccm89(newx, fmp['ebv'][j]*3.1, 3.1), model) #redden model
      model = model * x / (hplanck*1e7) / (clight*1e2) # erg/cm2/s/AA -> photons/cm2/s/AA

      #scale = median(model)/median(y[j,:])
      #print('scale1=',scale)

      x_brz =     concatenate((bx[(bx < min(rx))],
                        rx,
                        zx[(zx > max(rx))]))
      model_brz = concatenate((by['fit'][i,(bx < min(rx))],
                        ry['fit'][i,:],
                        zy['fit'][i,(zx > max(rx))]))

      model_brz = model_brz * 4. * pi  # Hlambda to Flambda

      scale = filter.get_flux(x_brz,model_brz).value /  10.**( 
                (fmp['gaia_phot_g_mean_mag'][j] + filter.Vega_zero_mag)/(-2.5) )

      #print('scale2=',scale)

      r = y[j,:]/model*scale
      w = ivar[j,:]*(model/scale)**2 
      if k == 0: 
        rr = r
        ww = w
      else:
        rr = vstack((rr,r))
        ww = vstack((ww,w))
      k += 1

  #plt.clf()
  #ma = mean(rr,0)            #straight mean response across spectra
  #ema = std(rr,0)/sqrt(k)    #uncertainty in mean
  mw = sum(ww*rr,0)/sum(ww,0)#weighted mean
  emw = 1./sqrt(sum(ww,0))   #uncertainty in w. mean
  me = median(rr,0)          #median 
  ms = smooth(mw,51)         #smoothed 
  mws = mw - ms              #scatter around the smoothed data
  ems = zeros(len(mw))
  length = 51
  for i in range(len(mw)): ems[i] = std(mws[max([0,i-length]):min([len(mws)-1,i+length])]) 

  #now we compute the relative fiber transmission (including flux losses due to centering)
  k = 0
  a = zeros(len(ind_sp))
  for i in ind_sp: 
      j = ind_sf[k]
      a[k] = mean(rr[k,:] / ms)
      print(i,j,a[k],fmp['fiber_ra'][j],fmp['fiber_dec'][j],fmp['gaia_phot_g_mean_mag'][j],mean(y[j,:]))
      k += 1

  print('n, median(emw/mw), median(ems/mw)=',k, median(emw/mw),median(ems/mw))
 

  return(x,mw,emw,ms,ems,a)


def calibrate(res,sframe):

  #observations
  sf=fits.open(sframe)
  fmp=sf['fibermap'].data
  x=sf['wavelength'].data
  y=sf['flux'].data
  ivar=sf['ivar'].data
  mask=sf['mask'].data
  resolution=sf['resolution'].data

  for j in range(len(fmp['fiber_ra'])):
    y[j,:] = y[j,:] / res
    y[j,:] = y[j,:] / x * (hplanck*1e7) * (clight*1e2) #   photons/cm2/s/AA -> erg/cm2/s/AA
    ivar[j,:] = ivar[j,:] * x**2 / (hplanck*1e7)**2 / (clight*1e2)**2
    y[j,:] = y[j,:] * 1e17
    ivar[j,:] = ivar[j,:] * 1e-34

  hdu0=fits.PrimaryHDU()
  now = datetime.datetime.fromtimestamp(time.time())
  nowstr = now.isoformat() 
  nowstr = nowstr[:nowstr.rfind('.')]
  hdu0.header['DATE'] = nowstr

  #get versions and enter then in primary header
  ver = get_versions()
  for entry in ver.keys(): hdu0.header[entry] = ver[entry]

  hdulist = [hdu0]

  npix = len(x)
  entry = sframe[7]
  print(entry,npix)

  hdu = fits.ImageHDU(name='WAVELENGTH', data=x)
  hdulist.append(hdu)
    
  hdu = fits.ImageHDU(name='FLUX', data=y)
  hdulist.append(hdu)

  hdu = fits.ImageHDU(name='IVAR', data=ivar)
  hdulist.append(hdu)

  hdu = fits.ImageHDU(name='MASK', data=mask)
  hdulist.append(hdu)

  hdu = fits.ImageHDU(name='RESOLUTION', data=resolution)
  hdulist.append(hdu)

  hdu=fits.BinTableHDU.from_columns(fmp, name='FIBERMAP')
  hdulist.append(hdu)

  hdul =fits.HDUList(hdulist)
  hdul.writeto('f'+sframe[1:])


  return None

#get dependencies versions, shamelessly copied from rvspec (Koposov's code)
def get_dep_versions():
    """
    Get Packages versions
    """
    packages = [
        'numpy', 'astropy', 'matplotlib', 'scipy',
        'yaml'
    ]
    # Ideally you need to check that the list here matches the requirements.txt
    ret = {}
    for curp in packages:
        ret[curp[:8]] = importlib.import_module(curp).__version__
    ret['python'] = str.split(sys.version, ' ')[0]
    return ret


#find out versions 
def get_versions():

  ver = get_dep_versions()
  ver['piferre'] = version
  log1file = glob.glob("*.log_01")
  fversion = 'unknown'
  if len(log1file) < 1:
    print("Warning: cannot find any *.log_01 file in the working directory")
  else:
    l1 = open(log1file[0],'r')
    #while 1:
    #  line = l1.readline()
    for line in l1:
      if 'f e r r e' in line:
        entries = line.split()
        fversion = entries[-1][1:]
        break
    l1.close()
    ver['ferre'] = fversion

  return(ver)

#get the maximum value of 'ellapsed time' (wall time) in all ferre std. output (log_*) files
def get_ferre_timings(proot):

  seconds = 0.0
  val = 0.
  logfiles=glob.glob(proot+'.log_*')
  for entry in logfiles:
    f=open(entry,'r')
    lines = tail(f,100)
    for line in lines:
      if 'ellapsed' in line:
        flds = line.split()
        val = float(flds[2])
    if val > seconds: seconds=val
    f.close()

  return(seconds)

def tail(f, lines=1, _buffer=4098):
#copied from https://stackoverflow.com/users/1889809/glenbot
    """Tail a file and get X lines from the end"""
    # place holder for the lines found
    lines_found = []

    # block counter will be multiplied by buffer
    # to get the block size from the end
    block_counter = -1

    # loop until we find X lines
    while len(lines_found) < lines:
        try:
            f.seek(block_counter * _buffer, os.SEEK_END)
        except IOError:  # either file is too small, or too many lines requested
            f.seek(0)
            lines_found = f.readlines()
            break

        lines_found = f.readlines()

        # we found enough lines, get out
        # Removed this line because it was redundant the while will catch
        # it, I left it for history
        # if len(lines_found) > lines:
        #    break

        # decrement the block counter to get the
        # next X bytes
        block_counter -= 1

    return lines_found[-lines:]

#get the time assigned in slurm for the calculation 
def get_slurm_timings(proot):

  seconds = nan
  f=open(proot+'.slurm','r')
  for line in f:
    if '--time' in line:
      entries = line.split()
      minutes = float(entries[1][7:])
      seconds = minutes*60.
      break

  return(seconds)

#get the number of cores assigned in slurm for the calculation 
def get_slurm_cores(proot):

  ncores = 0
  f=open(proot+'.slurm','r')
  for line in f:
    if '--cpus-per-task' in line:
      entries = line.split()
      ncores = int(entries[1][16:])
      break

  return(ncores)
  
#gather config. info
def load_conf(config='desi-n.yaml',confdir='.'):

  try:
    yfile=open(os.path.join(confdir,config),'r')
  except:
    print('ERROR in load_conf: cannot find the file ',config)
    return(None)
  #conf=yaml.full_load(yfile)
  conf=yaml.load(yfile, Loader=yaml.SafeLoader)
  yfile.close()

  return(conf)

#write piferre param. output
def write_tab_fits(root, path=None, config='desi-n.yaml'):
  
  conf=load_conf(config,confdir=confdir)

  if path is None: path=""
  proot=os.path.join(path,root)
  grids=conf['grids']
  v=glob.glob(proot+".vrd")
  o=glob.glob(proot+".opf")
  t=glob.glob(proot+".opt")
  #m=glob.glob(proot+".mdl")
  #n=glob.glob(proot+".nrd")
  if 'elem' in conf: 
    a=[]
    for entry in conf['elem']:
      a.append(proot+".oaf."+entry)

    prox=[]
    proxies=conf['proxy']
    for synth in grids:
      labels=conf[synth]['param']['labels']
      indproxies=zeros(len(proxies),dtype=int)
      j = 0
      for entry in proxies:
        i = 1
        for entry2 in labels:
          if entry == entry2: indproxies[j] = i
          i = i + 1
        j = j + 1
      prox.append(indproxies)


  fmp=glob.glob(proot+".fmp.fits")
  scr=glob.glob(proot+".scr.fits")
 

  if len(fmp) > 0:
    ff=fits.open(fmp[0])
    fibermap=ff[1]

  if len(scr) > 0:
    fs=fits.open(scr[0])
    scores=fs[1]
 
  success=[]
  targetid=[]
  target_ra=[]
  target_dec=[]
  ref_id=[]
  ref_cat=[]
  srcfile=[]
  bestgrid=[]
  teff=[]
  logg=[]
  feh=[]
  alphafe=[]
  micro=[]
  covar=[]
  elem=[]
  elem_err=[]
  snr_med=[]
  chisq_tot=[]
  rv_adop=[]
  rv_err=[]
  vf=open(v[0],'r')
  of=open(o[0],'r')
  if len(t) > 0: tf=open(t[0],'r')

  #set the stage for extracting abundances from oaf files
  if 'elem' in conf:
    af=[]
    i = 0
    for entry in conf['elem']: 
      af.append(open(a[i],'r'))
      i = i + 1


  for line in of:
    cells=line.split()
    k = int(cells[0])  # the very first line gives the index (1,2...) for the successful grid
    bestgrid.append(grids[k-1])
    cells = cells[1:]
    #for N dim (since COVPRINT=1 in FERRE), there are m= 4 + N*(2+N) cells
    #and likewise we can calculate N = sqrt(m-3)-1
    m=len(cells)
    assert (m > 6), 'Error, the file '+o[0]+' has less than 7 columns, which would correspond to ndim=2'
    ndim=int(sqrt(m-3)-1)
    cov = zeros((5,5)) #order is given by the 5-d kurucz grids ([Fe/H], [a/Fe], micro, Teff, logg)
    
    line = vf.readline()
    vcells=line.split()

    if len(t) > 0: 
      line = tf.readline()
      tcells=line.split()

    if (ndim == 2):
      #white dwarfs 2 dimensions: id, 2 par, 2err, 0., med_snr, lchi, 2x2 cov
      feh.append(-10.)
      if len(t) > 0:
        teff.append(float(tcells[1]))
      else:
        teff.append(float(cells[1]))
      logg.append(float(cells[2]))
      alphafe.append(nan)
      micro.append(nan)
      chisq_tot.append(10.**float(cells[3+2*ndim]))
      snr_med.append(float(cells[2+2*ndim]))
      rv_adop.append(float(vcells[6])*clight/1e3)
      rv_err.append(float(vcells[7])*clight/1e3)
      cov[3:,3:] = reshape(array(cells[4+2*ndim:],dtype=float),(2,2))
      covar.append(cov)    


    elif (ndim == 3):
      #Kurucz grids with 3 dimensions: id, 3 par, 3 err, 0., med_snr, lchi, 3x3 cov
      #see Allende Prieto et al. (2018, A&A)
      feh.append(float(cells[1]))
      if len(t) > 0:
        teff.append(float(tcells[2]))
      else:
        teff.append(float(cells[2]))
      logg.append(float(cells[3]))
      alphafe.append(nan)
      micro.append(nan)
      chisq_tot.append(10.**float(cells[3+2*ndim]))
      snr_med.append(float(cells[2+2*ndim]))
      rv_adop.append(float(vcells[6])*clight/1e3)
      rv_err.append(float(vcells[7])*clight/1e3)
      cov[2:,2:] = reshape(array(cells[4+2*ndim:],dtype=float),(3,3))
      cov[0,:] = cov[2,:]
      cov[2,:] = 0.
      cov[:,0] = cov[:,2]
      cov[:,2] = 0.
      covar.append(cov)

    elif (ndim == 4):
      #Phoenix grid from Sergey or MARCS grid, with 4 dimensions: id, 4 par, 4err, 0., med_snr, lchi, 4x4 cov
      feh.append(float(cells[2]))
      if len(t) > 0:
        teff.append(float(tcells[3]))
      else:
        teff.append(float(cells[3]))
      logg.append(float(cells[4]))
      alphafe.append(float(cells[1]))
      micro.append(nan)
      chisq_tot.append(10.**float(cells[3+2*ndim]))
      snr_med.append(float(cells[2+2*ndim]))
      rv_adop.append(float(vcells[6])*clight/1e3)
      rv_err.append(float(vcells[7])*clight/1e3)
      cov[1:,1:] = reshape(array(cells[4+2*ndim:],dtype=float),(4,4))
      cov[0,:] = cov[2,:]
      cov[2,:] = 0.
      cov[:,0] = cov[:,2]
      cov[:,2] = 0.
      covar.append(cov)    
   

    elif (ndim == 5):
      #Kurucz grids with 5 dimensions: id, 5 par, 5 err, 0., med_snr, lchi, 5x5 cov
      #see Allende Prieto et al. (2018, A&A)
      feh.append(float(cells[1]))
      if len(t) > 0:
        teff.append(float(tcells[4]))
      else:
        teff.append(float(cells[4]))
      logg.append(float(cells[5]))
      alphafe.append(float(cells[2]))
      micro.append(float(cells[3]))
      chisq_tot.append(10.**float(cells[3+2*ndim]))
      snr_med.append(float(cells[2+2*ndim]))
      rv_adop.append(float(vcells[6])*clight/1e3)
      rv_err.append(float(vcells[7])*clight/1e3)
      cov = reshape(array(cells[4+2*ndim:],dtype=float),(5,5))
      covar.append(cov)
     


    if (chisq_tot[-1] < 1.5 and snr_med[-1] > 5.): # chi**2<1.5 and S/N>5
      success.append(1) 
    else: success.append(0)
    tmp = cells[0].split('_')
    targetid.append(int64(tmp[0]))
    srcfile.append(root)
    #fiber.append(int32(tmp[1]))
    if 'elem' in conf:
      indproxies=prox[k-1]
      batch=[]
      batch_err=[]
      i = 0 
      for entry in conf['elem']:
        line = af[i].readline()
        acells = line.split()
        if '/H' in proxies[i]: 
          value=float(acells[indproxies[i]]) 
          value_err=float(acells[indproxies[i]+ndim])
        else:
          value=float(acells[indproxies[i]]) + feh[-1]
          value_err=sqrt(float(acells[indproxies[i]+ndim])**2 + covar[-1][0,0] )
    
        batch.append(value)
        batch_err.append(value_err)
        i = i + 1
      elem.append(batch)
      elem_err.append(batch_err)
    else:
      elem.append([nan,nan])
      elem_err.append([nan,nan])


  #add info copied from fibermap
  target_ra=fibermap.data['target_ra']
  target_dec=fibermap.data['target_dec']
  ref_id=fibermap.data['ref_id']
  ref_cat=fibermap.data['ref_cat']

  #primary extension
  hdu0=fits.PrimaryHDU()

  #find out processing date and add it to primary header
  now = datetime.datetime.fromtimestamp(time.time())
  nowstr = now.isoformat() 
  nowstr = nowstr[:nowstr.rfind('.')]
  hdu0.header['DATE'] = nowstr
  hdu0.header['FCONFIG'] = config

  #find out host machine and add info to header
  try:
    host=os.environ['HOST']
  except:
    host='Unknown'
  hdu0.header['HOST'] = host
  #find out OS name/platform
  osname = os.name 
  platf = platform.system() + ' '+ platform.release()
  hdu0.header['OS'] = osname
  hdu0.header['PLATFORM'] = platf

  #keep track of the number of targets processed and the time it took
  nspec = len(targetid)
  hdu0.header['NSPEC'] = nspec
  ftiming = get_ferre_timings(proot)
  hdu0.header['FTIME'] = ftiming
  stiming = get_slurm_timings(proot)
  hdu0.header['STIME'] = stiming
  ncores = get_slurm_cores(proot)
  hdu0.header['NCORES'] = ncores
  global_conf=dict(conf['global'])
  hdu0.header['NTHREADS'] = global_conf['nthreads']

  #get versions and enter then in primary header
  ver = get_versions()
  for entry in ver.keys(): hdu0.header[entry] = ver[entry]

  
  hdulist = [hdu0]

  #sptab extension
  cols = {}
  cols['SUCCESS'] = success
  cols['TARGETID'] = targetid
  cols['TARGET_RA'] = target_ra
  cols['TARGET_DEC'] = target_dec
  cols['REF_ID'] = ref_id
  cols['REF_CAT'] = ref_cat
  cols['SRCFILE'] = srcfile
  cols['BESTGRID'] = bestgrid
  cols['TEFF'] = array(teff)*units.K
  cols['LOGG'] = array(logg)
  cols['FEH'] = array(feh)
  cols['ALPHAFE'] = array(alphafe) 
  cols['LOG10MICRO'] = array(micro)
  cols['PARAM'] = vstack ( (feh, alphafe, micro, teff, logg) ).T
  cols['COVAR'] = array(covar).reshape(len(success),5,5)
  cols['ELEM'] = array(elem)
  cols['ELEM_ERR'] = array(elem_err)
  cols['CHISQ_TOT'] = array(chisq_tot)
  cols['SNR_MED'] = array(snr_med)
  cols['RV_ADOP'] = array(rv_adop)*units.km/units.s
  cols['RV_ERR'] = array(rv_err)*units.km/units.s

  colcomm = {
  'success': 'Bit indicating whether the code has likely produced useful results',
  'TARGETID': 'DESI targetid',
  'TARGET_RA': 'Target Right Ascension (deg) -- details in FIBERMAP',
  'TARGET_DEC': 'Target Declination (deg) -- details in FIBERMAP',
  'REF_ID': 'Astrometric cat refID (Gaia SOURCE_ID)',
  'REF_CAT': 'Astrometry reference catalog',
  'SRCFILE': 'DESI data file',
  'BESTGRID': 'Model grid that produced the best fit',
  'TEFF': 'Effective temperature (K)',
  'LOGG': 'Surface gravity (g in cm/s**2)',
  'FEH': 'Metallicity [Fe/H] = log10(N(Fe)/N(H)) - log10(N(Fe)/N(H))sun' ,
  'ALPHAFE': 'Alpha-to-iron ratio [alpha/Fe]',
  'LOG10MICRO': 'Log10 of Microturbulence (km/s)',
  'PARAM': 'Array of atmospheric parameters ([Fe/H], [a/Fe], log10micro, Teff,logg)',
  'COVAR': 'Covariance matrix for ([Fe/H], [a/Fe], log10micro, Teff,logg)',
  'ELEM': 'Elemental abundance ratios to hydrogen [elem/H]',
  'ELEM_ERR': 'Uncertainties in the elemental abundance ratios',
  'CHISQ_TOT': 'Total chi**2',
  'SNR_MED': 'Median signal-to-ratio',
  'RV_ADOP': 'Adopted Radial Velocity (km/s)',
  'RV_ERR': 'Uncertainty in the adopted Radial Velocity (km/s)'
  }      

  
  table = tbl.Table(cols)
  hdu=fits.BinTableHDU(table,name = 'SPTAB')
  #hdu.header['EXTNAME']= ('SPTAB', 'Stellar Parameter Table')
  k = 0
  for entry in colcomm.keys():
    print(entry) 
    hdu.header['TCOMM'+"{:03d}".format(k+1)] = colcomm[entry]
    k+=1
  hdulist.append(hdu)

  #fibermap extension
  if len(fmp) > 0:
    hdu=fits.BinTableHDU.from_columns(fibermap, name='FIBERMAP')
    hdulist.append(hdu)
    ff.close()

  #scores extension
  if len(scr) > 0:
    hdu=fits.BinTableHDU.from_columns(scores, name='SCORES')
    hdulist.append(hdu)
    fs.close()

  #aux extension
  p = ['[Fe/H]','[a/Fe]','log10micro','Teff','logg']
  if 'elem' in conf: e = conf['elem']
  cols = {}
  colcomm = {}
  cols['p'] = [p]
  colcomm['p'] = 'PARAM tags'
  if 'elem' in conf:
    cols['e'] = [e]
    colcomm['e']= 'ELEM tags'
  #cols['ip']= [dict(zip(p,arange(len(p))))]
  #colcomm['ip']= 'Indices for PARAM tags'
  #if 'elem' in conf:
  #  cols['ie']= [dict(zip(e,arange(len(e))))]
  #  colcomm['ie']= 'Indices for ELEM tags'
  
  table = tbl.Table(cols)
  hdu=fits.BinTableHDU(table,name = 'AUX')

  k = 0
  for entry in colcomm.keys():
    print(entry) 
    hdu.header['TCOMM'+"{:03d}".format(k+1)] = colcomm[entry]
    k+=1
  hdulist.append(hdu)


  hdul=fits.HDUList(hdulist)
  hdul.writeto('sptab_'+root+'.fits')
  
  return None

#write piferre spec. output  
def write_mod_fits(root, path=None, config='desi-n.yaml'):  
  
  if path is None: path=""
  proot=os.path.join(path,root)

  #gather config. info
  conf=load_conf(config,confdir=confdir)
  global_conf=dict(conf['global'])

  
  xbandfiles = sorted(glob.glob(proot+'-*.wav'))
  band = []
  npix = []
  for entry in xbandfiles:
    match = re.search('-[\w]*.wav',entry)
    tag = match.group()[1:-4]
    if match: band.append(tag.upper())
    x = loadtxt(proot+'-'+tag+'.wav')
    npix.append(len(x))
    
  x = loadtxt(proot+'.wav')
  if len(npix) == 0: npix.append(len(x))

  m=glob.glob(proot+".mdl")
  e=glob.glob(proot+".err")
  n=glob.glob(proot+".nrd")
  l=glob.glob(proot+".ndl")
  a=glob.glob(proot+".nal.*")

  fmp=glob.glob(proot+".fmp.fits")  
  scr=glob.glob(proot+".scr.fits")
  edata=loadtxt(e[0])
  if (len(m) > 0): 
    mdata=loadtxt(m[0])
  if (len(n) > 0): 
    odata=loadtxt(n[0])
    f=glob.glob(proot+".frd")
    fdata=loadtxt(f[0])
    edata=edata/fdata*odata
  else:
    odata=loadtxt(proot+".frd")  

  if (len(l) > 0): 
    ldata=loadtxt(l[0])

  if ('elem' in conf and len(a) > 0): 
    i = 0
    for entry in conf['elem']:
      filterfile=conf['extensions']['abund']['filterfile']
      if '$elem' in filterfile: 
        filterfile = filterfile.replace('$elem',str(entry))
      #if '$synth' in filterfile: 
      #  filterfile = filterfile.replace('$synth',str('fillmein-please'))
      filt=loadtxt(os.path.join(filterdir,filterfile))
      farr=loadtxt(proot+".nal."+entry)
      if i == 0:  
        w=filt < 1e-4
        if farr.ndim == 2:
          adata=farr[:,:]
          adata[:,w]=0.0
        else:
          adata=farr[:]
          adata[w]=0.0
      else:
        w=filt >= 1e-4
        if farr.ndim == 2:
          adata[:,w] = adata[:,w] + farr[:,w]
        else:
          adata[w] = adata[w] + farr[w]
      i = i + 1


  hdu0=fits.PrimaryHDU()
  now = datetime.datetime.fromtimestamp(time.time())
  nowstr = now.isoformat() 
  nowstr = nowstr[:nowstr.rfind('.')]
  hdu0.header['DATE'] = nowstr
  hdu0.header['FCONFIG'] = config

  #find out host machine and add info to header
  try:
    host=os.environ['HOST']
  except:
    host='Unknown'
  hdu0.header['HOST'] = host
  #find out OS name/platform
  osname = os.name 
  platf = platform.system() + ' '+ platform.release()
  hdu0.header['OS'] = osname
  hdu0.header['PLATFORM'] = platf

  #keep track of the number of targets processed and the time it took
  nspec = len(odata)
  hdu0.header['NSPEC'] = nspec
  ftiming = get_ferre_timings(proot)
  hdu0.header['FTIME'] = ftiming
  stiming = get_slurm_timings(proot)
  hdu0.header['STIME'] = stiming
  ncores = get_slurm_cores(proot)
  hdu0.header['NCORES'] = ncores
  hdu0.header['NTHREADS'] = global_conf['nthreads']

  #get versions and enter then in primary header
  ver = get_versions()
  for entry in ver.keys(): hdu0.header[entry] = ver[entry]


  hdulist = [hdu0]

  i = 0
  j1 = 0


  for entry in band:
    j2 = j1 + npix[i] 
    print(entry,i,npix[i],j1,j2)
    #colx = fits.Column(name='wavelength',format='e8', array=array(x[j1:j2]))
    #coldefs = fits.ColDefs([colx])
    #hdu = fits.BinTableHDU.from_columns(coldefs)
    hdu = fits.ImageHDU(name=entry+'_WAVELENGTH', data=x[j1:j2])
    hdulist.append(hdu)
    
    cols = {}
    colcomm = {}
    if odata.ndim == 2: tdata = odata[:,j1:j2]
    else: tdata = odata[j1:j2][None,:]
    cols['obs'] = tdata
    colcomm['obs'] = 'Observed spectra as fit'
    if edata.ndim == 2: tdata = edata[:,j1:j2]
    else: tdata = edata[j1:j2][None,:]
    cols['err'] = tdata
    colcomm['err'] = 'Error in spectra as fit'
    if (len(m) > 0): 
      if mdata.ndim == 2: tdata = mdata[:,j1:j2]
      else: tdata = mdata[j1:j2][None,:]
      cols['flx'] = tdata
      colcomm['flx'] = 'Absolute flux for best-fitting model'
    if (len(l) > 0): 
      if ldata.ndim == 2: tdata = ldata[:,j1:j2]
      else: tdata = ldata[j1:j2][None,:]
      cols['fit'] = tdata
      colcomm['fit'] = 'Best-fitting model (atmospheric parameters)'
    if ('elem' in conf and len(a) > 0):
      if adata.ndim == 2: tdata = adata[:,j1:j2]
      else: tdata = adata[j1:j2][None,:]
      cols['abu'] = tdata
      colcomm['abu'] = 'Best-fitting model (abundances)'
      
      

    table = tbl.Table(cols)
    hdu=fits.BinTableHDU(table,name = entry+'_MODEL')
    k = 0
    for entry in colcomm.keys():
      print(entry) 
      hdu.header['TCOMM'+"{:03d}".format(k+1)] = colcomm[entry]
      k+=1
    hdulist.append(hdu)
    i += 1
    j1 = j2

  if len(fmp) > 0:
    ff=fits.open(fmp[0])
    fibermap=ff[1]
    hdu=fits.BinTableHDU.from_columns(fibermap, name='FIBERMAP')
    #hdu.header['EXTNAME']='FIBERMAP'
    hdulist.append(hdu)

  if len(scr) > 0:
    ff=fits.open(scr[0])
    scores=ff[1]
    hdu=fits.BinTableHDU.from_columns(scores, name='SCORES')
    hdulist.append(hdu)

  #FILTER extension
  if 'elem' in conf:
    e = conf['elem']
    cols = {}
    colcomm = {}
    for entry in e:
      filterfile=conf['extensions']['abund']['filterfile']
      if '$elem' in filterfile: 
        filterfile = filterfile.replace('$elem',str(entry))
      #if '$synth' in filterfile: 
      #  filterfile = filterfile.replace('$synth',str('fillmein-please'))
      filt=loadtxt(os.path.join(filterdir,filterfile))
      cols[entry] = filt
      colcomm[entry] = entry+' FILTER'
  
    table = tbl.Table(cols)
    hdu=fits.BinTableHDU(table,name = 'FILTER')

    k = 0
    for entry in colcomm.keys():
      print(entry) 
      hdu.header['TCOMM'+"{:03d}".format(k+1)] = colcomm[entry]
      k+=1
    hdulist.append(hdu)


  hdul=fits.HDUList(hdulist)
  hdul.writeto('spmod_'+root+'.fits')
  
  return None

#write ferre files
def write_ferre_input(root,ids,par,y,ey,path=None):

  if path is None: path="./"

  #open ferre input files
  vrd=open(os.path.join(path,root)+'.vrd','w')
  frd=open(os.path.join(path,root)+'.frd','w')
  err=open(os.path.join(path,root)+'.err','w')

  nspec, freq = y.shape

  #loop to write data files
  i=0
  while (i < nspec):

    if i%10 == 0: print('. ',end='',flush=True)
    if i%100 == 0 : print(str(i),end='',flush=True)

    #vrd.write("target_"+str(i+1)+" 0.0 0.0 0.0")
    ppar=[ids[i]]
    for item in par[ids[i]]: ppar.append(item)
    #vrd.write(' '.join(map(str,ppar))
    vrd.write("%30s %6.2f %10.2f %6.2f %6.2f %12.9f %12.9f %12.9f %12.9f %12.9f\n" % 
tuple(ppar) )    
    #ppar.tofile(ppar,sep=" ",format="%s")
    #vrd.write("\n")

    yy=y[i,:]
    yy.tofile(frd,sep=" ",format="%0.4e")
    frd.write("\n")
    eyy=ey[i,:]
    eyy.tofile(err,sep=" ",format="%0.4e")
    err.write("\n")
    i+=1
    #print(i,yy[0],eyy[0])
 
  #close files
  vrd.close()
  frd.close()
  err.close()
  print('/')

def opfmerge(root,path=None,wait_on_sorted=False,config='desi-n.yaml'):

  if path is None: path="./"
  proot=os.path.join(path,root)

  if wait_on_sorted:
    o=sorted(glob.glob(proot+".opf*_sorted"))  
    while (len(o) > 0):
      time.sleep(5)
      o=sorted(glob.glob(proot+".opf*_sorted"))
      

  o=sorted(glob.glob(proot+".opf?*"))
  m=sorted(glob.glob(proot+".mdl?*"))
  n=sorted(glob.glob(proot+".nrd?*"))
  l=sorted(glob.glob(proot+".ndl?*"))
  t=sorted(glob.glob(proot+".opt?*"))
 

  llimit=[] # lower limits for Teff
  iteff=[]  # column for Teff in opf
  ilchi=[]  # column for log10(red. chi**2) in opf
  conf=load_conf(config,confdir=confdir)  
  #set the set of grids to be used
  grids=conf['grids']
  for entry in grids:
    tmplist=conf[entry]['param']['labels']
    iteffcol = [idx for idx, element in enumerate(tmplist) if element == 'Teff'][0]
    tmpstr=conf[entry]['param']['llimits']
    tmplist=tmpstr.split()
    llimit.append(float(tmplist[iteffcol]))
    iteff.append(iteffcol+1)
    ilchi.append(conf[entry]['param']['ndim']*2+3)


  ngrid=len(o)
  if (len(m) > 0):
    if ngrid != len(m): 
      print("there are different number of opf?* and mdl?* arrays")
      return(0)
  if (len(n) > 0):
    if ngrid != len(n):  
      print("there are different number of opf?* and nrd?* arrays")
      return(0)
  if (len(l) > 0):
    if ngrid != len(l):  
      print("there are different number of opf?* and ndl?* arrays")
      return(0)
  if (len(t) > 0):
    if ngrid != len(t):  
      print("there are different number of opf?* and opt?* arrays")
      return(0)


  #open input files
  of=[]
  if len(m) > 0: mf=[]
  if len(n) > 0: nf=[]
  if len(l) > 0: lf=[]
  if len(t) > 0: tf=[]
  for i in range(len(o)):
    of.append(open(o[i],'r'))
    if len(m) > 0: mf.append(open(m[i],'r'))
    if len(n) > 0: nf.append(open(n[i],'r'))
    if len(l) > 0: lf.append(open(l[i],'r'))
    if len(t) > 0: tf.append(open(t[i],'r'))
  print(o)
  print(of)
  #open output files
  oo=open(proot+'.opf','w')
  if len(m) > 0: mo=open(proot+'.mdl','w')
  if len(n) > 0: no=open(proot+'.nrd','w')
  if len(l) > 0: lo=open(proot+'.ndl','w')
  if len(t) > 0: to=open(proot+'.opt','w')
 
  for line in of[0]: 
    tmparr=line.split()
    min_chi=float(tmparr[ilchi[0]])
    min_oline=line
    print(min_chi,min_oline)
    if len(m) > 0: min_mline=mf[0].readline()
    if len(n) > 0: min_nline=nf[0].readline()
    if len(l) > 0: min_lline=lf[0].readline()
    if len(t) > 0: min_tline=tf[0].readline()
    k = 0
    for i in range(len(o)-1):
      oline=of[i+1].readline()
      if len(m) > 0: mline=mf[i+1].readline()
      if len(n) > 0: nline=nf[i+1].readline()
      if len(l) > 0: lline=lf[i+1].readline()
      if len(t) > 0: tline=tf[i+1].readline()
      tmparr=oline.split()
      #print(len(tmparr))
      #print(tmparr)
      #print(i,ilchi[i+1],len(tmparr))
      #print(i,float(tmparr[ilchi[i+1]]))
      if float(tmparr[ilchi[i+1]]) < min_chi and float(tmparr[iteff[i+1]]) > llimit[i+1]*1.01: 
        min_chi=float(tmparr[ilchi[i+1]])
        min_oline=oline
        if len(m) > 0: min_mline=mline
        if len(n) > 0: min_nline=nline
        if len(l) > 0: min_lline=lline
        if len(t) > 0: min_tline=tline
        k = i + 1
    
    #print(min_chi,min_oline)
    oo.write("{:02d}".format(k+1)+' '+min_oline)
    if len(m) > 0: mo.write(min_mline)
    if len(n) > 0: no.write(min_nline)
    if len(l) > 0: lo.write(min_lline)
    if len(t) > 0: to.write(min_tline)
  
  #close input files
  for i in range(len(o)):
    #print(o[i],m[i])
    of[i].close
    if len(m) > 0: mf[i].close
    if len(n) > 0: nf[i].close
    if len(l) > 0: lf[i].close
    if len(t) > 0: tf[i].close

  #close output files
  oo.close
  if len(m) > 0: mo.close
  if len(n) > 0: no.close
  if len(l) > 0: lo.close
  if len(t) > 0: to.close
  
  return None

def oafmerge(root,path=None,wait_on_sorted=False,config='desi-n.yaml'):

  if path is None: path="./"
  proot=os.path.join(path,root)

  if wait_on_sorted:
    a=sorted(glob.glob(proot+".oaf*_sorted"))  
    while (len(o) > 0):
      time.sleep(5)
      a=sorted(glob.glob(proot+".oaf*_sorted"))
      

  conf=load_conf(config,confdir=confdir)

  if 'elem' not in conf: return None

  #set the set of grids to be used
  grids=conf['grids']
  elem=conf['elem']
  abund_grids=conf['abund_grids']
  indices=[]
  for agr in abund_grids:
    i=1
    for gr in grids:
      if gr == agr: indices.append(i)
      i = i + 1

  #setup a line to includ in the oaf files when no abundance has
  #been derived for the winning grid
  of=open(proot+'.opf')
  line=of.readline()
  of.close()
  cols=line.split()
  ncol=len(cols)
  bads=zeros(ncol)
  bads[:]=nan
  badline=' '.join(map(str,bads))+'\n'
  of=open(proot+'.frd')
  line=of.readline()
  of.close()
  cols=line.split()
  ncol=len(cols)
  bads=zeros(ncol)
  bads[:]=nan
  longbadline=' '.join(map(str,bads))+'\n'
  


  for el in elem:
    a=[]
    for en in indices:
      a.append(proot+".oaf."+el+"{:02d}".format(en))

    #open input files
    of=open(proot+".opf")
    af=[]
    df=[]
    lf=[]
    for i in range(len(a)):
      print(a[i],a[i].replace('oaf','nad'))
      af.append(open(a[i],'r'))
      df.append(open(a[i].replace('oaf','nad'),'r'))
      lf.append(open(a[i].replace('oaf','nal'),'r'))
    #open output files
    ao=open(proot+'.oaf.'+el,'w')
    do=open(proot+'.nad.'+el,'w')
    lo=open(proot+'.nal.'+el,'w')
 
    for line in of: 
      tmparr=line.split()
      k = int(tmparr[0])
      gotit=False         
      for i in range(len(a)):
        aline=af[i].readline()
        dline=df[i].readline()
        lline=lf[i].readline()
        if (k == indices[i]): 
          gotit=True
          ao.write(aline)
          do.write(dline)
          lo.write(lline)
      if not gotit: 
          ao.write(badline)
          do.write(longbadline)
          lo.write(longbadline) 
     

    #close input files
    of.close()
    for i in range(len(a)):
      af[i].close()
      df[i].close()
      lf[i].close()

    #close output files
    ao.close()
    do.close()
    lo.close()
  
  return None


#get names of extensions from a FITS file
def extnames(hdu):
  #hdu must have been open as follows hdu=fits.open(filename)
  x=hdu.info(output=False)
  names=[]
  for entry in x: names.append(entry[1])
  return(names)

#run
def run(pixel,path=None):
  if path is None: path="./"
  #directly run
  #pwd=os.path.abspath(os.curdir)
  #os.chdir(path)
  #job="/bin/bash "+pixel+'.slurm'
  #code=subprocess.call(job)
  #os.chdir(pwd)
  #slurm
  job="sbatch "+os.path.join(path,pixel+'.slurm')
  code=subprocess.call(job)
  #kiko
  #job="kiko "+os.path.join(path,pixel+'.slurm')
  #code=subprocess.call(job)

  return code

#find pixels in 'root' directory (spectra-64)
def getpixels(root):
  #root='spectro/redux/dc17a2/spectra-64/'
  d1=os.listdir(root)
  d=[]
  for x in d1:
    #print('x=',x)
    assert os.path.isdir(os.path.join(root,x)), 'the data directory must contain folders, not data files'
    d2=os.listdir(os.path.join(root,x))
    #d.append(os.path.join(root,x))
    res=[i for i in d2 if '.fits' in i] 
    for y in d2: 
      #d.append(os.path.join(root,x))
      if len(res) == 0: # there are no fits files in the 1st directory, so 2 layer (e.g. DESI)
        d.append(os.path.join(root,x,y))
      else: 
        entry=os.path.join(root,x)
        if entry not in d: d.append(entry)  #keep only the first layer (SDSS/BOSS)

  print('found pixels: ',d)
  print(len(d))
  return(d)

#get spectra and matching RV files
def getdata(sppath='.',rvpath=None,sptype='spectra',rvtype='zbest'):

  if rvpath is None: rvpath = sppath

  spfiles1 = list(glob.iglob(os.path.join(sppath,'**',sptype+'*fits'), recursive=True))
  rvfiles1 = list(glob.iglob(os.path.join(rvpath,'**',rvtype+'*fits'), recursive=True))
  
  print('spfiles1=',spfiles1)
  print('rvfiles1',rvfiles1)

  spfiles = []
  rvfiles = []
  for entry in spfiles1:
    filename = os.path.split(entry)[-1]
    #print('filename=',filename,' pattern=',filename[len(sptype):])
    entry2 = list(filter(lambda x: filename[len(sptype):] in x, rvfiles1))
    if len(entry2) == 1:
      spfiles.append(entry)
      rvfiles.append(entry2[0])
    elif len(entry2) < 1:
      print('Warning: there is no matching rv file for ',entry,' -- we skip this file')
    else:
      print('Warning: there are multiple matching rv files for ',entry,' -- we skip this file')
      print('         rv matching files:',' '.join(entry2))
  
  #analyze the situation wrt input files
  nsp=len(spfiles)
  nrv=len(rvfiles)

  print ('Found '+str(nsp)+' input spectra files')
  for filename in spfiles: print(filename+'--')
  print ('and '+str(nrv)+' associated rv files')
  for filename in rvfiles: print(filename+'--')

  if (nsp != nrv):
    print('ERROR -- there is a mismatch between the number of spectra files and rv files, this pixel is skipped')
    return (None,None)

  return (spfiles,rvfiles)


#identify input data files and associated zbest files 
#obsolete, use getdata instead
def finddatafiles(path,pixel,sdir='',rvpath=None):

  if rvpath is None: rvpath = path

  print(path,rvpath)
  print('path,sdir,pixel=',path,sdir,pixel)

  infiles=os.listdir(os.path.join(path,sdir,pixel))  
  datafiles=[]
  zbestfiles=[]
  for ff in infiles: #add subdirs, which may contain zbest files for SDSS/BOSS
    if os.path.isdir(os.path.join(path,sdir,pixel,ff)): 
      extrafiles=os.listdir(os.path.join(path,sdir,pixel,ff))
      for ff2 in extrafiles: 
        infiles.append(os.path.join(ff,ff2))

  print(path,rvpath)
  if rvpath != path:
    print('hola')
    if os.path.isdir(os.path.join(rvpath,sdir,pixel)):
      infiles2=os.listdir(os.path.join(rvpath,sdir,pixel))
      for ff in infiles2: #add subdirs, which may contain zbest files for SDSS/BOSS
        infiles.append(ff)
        if os.path.isdir(os.path.join(rvpath,sdir,pixel,ff)): 
          extrafiles2=os.listdir(os.path.join(rvpath,sdir,pixel,ff))
          for ff2 in extrafiles2: 
            infiles.append(os.path.join(ff,ff2))


  infiles.sort()
  print('infiles=',infiles)

  for filename in infiles:
# DESI sims/data
    if (filename.find('spectra-') > -1 and filename.find('.fits') > -1):
      datafiles.append(os.path.join(path,sdir,pixel,filename))
    elif (filename.find('zbest-') > -1 and filename.find('.fits') > -1):
      zbestfiles.append(os.path.join(rvpath,sdir,pixel,filename))
    elif (filename.find('redrock-') > -1 and filename.find('.fits') > -1):
      zbestfiles.append(os.path.join(rvpath,sdir,pixel,filename))
# BOSS data
    elif (filename.find('spPlate') > -1 and filename.find('.fits') > -1):
      datafiles.append(os.path.join(path,sdir,pixel,filename))
    elif (filename.find('spZbest') > -1 and filename.find('.fits') > -1):
      zbestfiles.append(os.path.join(rvpath,sdir,pixel,filename))
#  DESI RVSPEC files
    elif (filename.find('rvtab') > -1 and filename.find('.fits') > -1):
      zbestfiles.append(os.path.join(rvpath,sdir,pixel,filename))


  #analyze the situation wrt input files
  ndatafiles=len(datafiles)
  nzbestfiles=len(zbestfiles)
  print ('Found '+str(ndatafiles)+' input data files')
  for filename in datafiles: print(filename+'--')
  print ('and '+str(nzbestfiles)+' associated zbest files')
  for filename in zbestfiles: print(filename+'--')

  if (ndatafiles != nzbestfiles):
    print('ERROR -- there is a mismatch between the number of data files and zbest files, this pixel is skipped')
    return (None,None)

  return (datafiles,zbestfiles)


#pack a collection of fits files with binary tables in multiple HDUs into a single one
def packfits(input="*.fits",output="output.fits",update_srcfile=False):


  f = sorted(glob.glob(input))

  print('reading ... ',f[0])
  hdul1 = fits.open(f[0])
  hdu0 = hdul1[0]
  for entry in f[1:]:
    hdulist = [hdu0]       
    print('reading ... ',entry)
    hdul2 = fits.open(entry)
    for i in arange(len(hdul1)-1)+1:
      nrows1 = hdul1[i].data.shape[0]
      nrows2 = hdul2[i].data.shape[0]
      nrows = nrows1 + nrows2
      if (str(type(hdul1[i])) == "<class 'astropy.io.fits.hdu.table.BinTableHDU'>"): #binary tables
        hdu = fits.BinTableHDU.from_columns(hdul1[i].columns, nrows=nrows)
        hdu.header['EXTNAME'] = hdul1[i].header['EXTNAME']
        if (str(type(hdul2[i])) != "<class 'astropy.io.fits.hdu.table.BinTableHDU'>"): 
          print(i, str(type(hdul1[i])),str(type(hdul2[i])))
          print('Warning: the extension ',i, 'in file ',entry,' is not a binary table as expected based on the preceding files. The extension is skipped.')

        for colname in hdul1[i].columns.names:
          #print('adding colname=',colname,' from the 2nd file')
          try:
            if colname in hdul2[i].columns.names:
              #print(hdul1[i].data[colname].shape,hdul2[i].data[colname].shape)
              #print(hdu.data[colname].shape)
              hdu.data[colname][nrows1:] = hdul2[i].data[colname]
            else: print('Warning: the file ',entry,' does not include column ',colname,' in extension ',i,' -- ',hdu.header['EXTNAME'])
          except AttributeError:
            print('Warning: the file ',entry,' does not have the attribute columns ',' in extension ',i,' -- ',hdu.header['EXTNAME']) 
          if update_srcfile and colname == 'SRCFILE':
            if entry == f[1]: 
              hdu.data[colname][:nrows1] = '-'.join(f[0].split(os.path.sep)[-1].split('-')[1:])
            hdu.data[colname][nrows1:] = '-'.join(entry.split(os.path.sep)[-1].split('-')[1:])
            print(len(hdu.data[colname][nrows1]),len(entry.split(os.path.sep)[-1]))

              

      elif (str(type(hdul1[i])) == "<class 'astropy.io.fits.hdu.image.ImageHDU'>"): #images
        hdu = fits.PrimaryHDU(vstack( (hdul1[i].data, hdul2[i].data) ))
        hdu.header['EXTNAME'] = hdul1[i].header['EXTNAME']

      hdulist.append(hdu) 

    hdul1 = fits.HDUList(hdulist)

  hdul1.writeto(output)

  return(None)
  
#calls packfits hierarchically in a folder tree
#to pack the results from a DESI data tree
#e.g. try calling it from healpix with structure like
#healpix/cmx/backup/gpix/hpix
def treepackspfits(input='sptab*.fits',path='./',depth=3):
  sites1 = [] #deepest layer, gpix (where srcfile will be updated)
  sites2 = [] #rest of layers
  base_depth = path.rstrip(os.path.sep).count(os.path.sep)
  for root, dirs, files in os.walk(path,topdown=False):
    for entry in dirs:
      cur_depth = os.path.join(root,entry).count(os.path.sep) 
      print(os.path.join(root,entry),cur_depth)
      if base_depth + depth == cur_depth:
        sites1.append(os.path.join(root,entry))
      elif base_depth + depth > cur_depth:
        sites2.append(os.path.join(root,entry))
      
  sites2.append(path)
  
  sites = sites1 + sites2
  
  i = 0
  startpath = os.path.abspath(os.curdir)
  for entry in sites:
    i = i + 1
    os.chdir(startpath)
    os.chdir(entry)
    infiles = glob.glob(os.path.join('*',input))
    print('pwd=',os.path.abspath(os.curdir))
    print('infiles=',infiles)
    if len(infiles) < 1: continue
    parts = infiles[0].split(os.path.sep)[1].split('-')
    ext = parts[-1].split('.')[-1]
    
    if entry in sites1:
      #drop petal info from filename, since petals are merged
      if parts[1] in list(map(str,range(10))): parts.pop(1) 
      outfile = '-'.join(parts[:-1]) + '-' + entry.split(os.path.sep)[-1] + '.' + ext
      packfits(os.path.join('*',input),output=outfile,update_srcfile=True)
    else:
      outfile = '-'.join(parts[:-1]) + '.' + ext
      packfits(os.path.join('*',input),output=outfile,update_srcfile=False)			
 			
  return(None)
  
#inspector
def inspect(sptabfile,sym='.',rvrange=(-1e32,1e32),
               rarange=(0.,360.), decrange=(-90,90),  
               pmrarange=(-1e10,1e10), pmdecrange=(-1e10,1e10),
               parallaxrange=(-10000,10000),
               fehrange=(-100,100), teffrange=[0,100000],loggrange=[-100,100],
               chisq_totrange=(0.0,1e32),snr_medrange=(0,1e32),
               title='',fig='',plot=True):

    sph=fits.open(sptabfile)
    spt=sph['SPTAB'].data
    fbm=sph['FIBERMAP'].data
    w=( (spt['rv_adop'] >= rvrange[0])  
        & (spt['rv_adop'] <= rvrange[1])   
        & (spt['feh'] >= fehrange[0])
        & (spt['feh'] <= fehrange[1]) 
        & (spt['teff'] >= teffrange[0]) 
        & (spt['teff'] <= teffrange[1])
        & (spt['logg'] >= loggrange[0])
        & (spt['logg'] <= loggrange[1]) 
        & (spt['chisq_tot'] >= chisq_totrange[0])
        & (spt['chisq_tot'] <= chisq_totrange[1]) 
        & (spt['snr_med'] >= snr_medrange[0])
        & (spt['snr_med'] <= snr_medrange[1]) 
        & (fbm['target_ra'] >= rarange[0])      
        & (fbm['target_ra'] <= rarange[1])
        & (fbm['target_dec'] >= decrange[0]) 
        & (fbm['target_dec'] <= decrange[1]) 
        & (fbm['parallax'] >= parallaxrange[0])
        & (fbm['parallax'] <= parallaxrange[1]) 
        & (fbm['pmra'] >= pmrarange[0])
        & (fbm['pmra'] <= pmrarange[1]) 
        & (fbm['pmdec'] >= pmdecrange[0])
        & (fbm['pmdec'] <= pmdecrange[1]) )

    spt=spt[w]
    fbm=fbm[w]
    n=where(w)[0]

    if plot:
        plt.figure()
        plt.tight_layout()
        plt.ion()

        plt.subplot(3,2,1)
        plt.plot(spt['feh'],spt['alphafe'],sym)
        plt.xlabel('[Fe/H]',labelpad=0)
        plt.ylabel('[a/Fe]')
        plt.title(sptabfile)

        plt.subplot(3,2,2)
        plt.hist(spt['teff'],bins=50)
        plt.xlabel('Teff',labelpad=0)
        plt.ylabel('N')
        plt.title(title)

        plt.subplot(3,2,3)
        plt.plot(spt['teff'],spt['logg'],sym)
        plt.xlabel('Teff',labelpad=0)
        plt.ylabel('logg')
        plt.xlim([8000.,2000])
        plt.ylim([5.5,-0.5])

        plt.subplot(3,2,4)
        plt.plot(spt['teff'],spt['feh'],sym)
        plt.xlabel('Teff',labelpad=0)
        plt.ylabel('[Fe/H]')
        plt.xlim([8000.,2000])
        plt.ylim([-5,1])


        plt.subplot(3,2,5)
        plt.hist(spt['rv_adop'],bins=50)
        plt.xlabel('RV',labelpad=0)
        plt.ylabel('N')


        plt.subplot(3,2,6)
        plt.hist(spt['feh'],bins=50)
        plt.xlabel('[Fe/H]',labelpad=0)
        plt.ylabel('N')
        me = median(spt['feh'])
        mm = mean(spt['feh'])
        ss = std(spt['feh'])
        plt.text(me-2*ss, 1, r'$\mu=$'+"{:5.2f}".format(mm) )
        plt.text(me+2*ss, 1, r'$\sigma=$'+"{:5.2f}".format(ss) )


        if fig != '': plt.savefig(fig)

        plt.show()

    return (spt,fbm)

#inspector
def inspect3(sptabfile,sym='.',rvrange=(-1e32,1e32),
               rarange=(0.,360.), decrange=(-90,90),  
               pmrarange=(-1e10,1e10), pmdecrange=(-1e10,1e10),
               parallaxrange=(-10000,10000),
               fehrange=(-100,100), teffrange=[0,100000],loggrange=[-100,100],
               chisq_totrange=(0.0,1e32),snr_medrange=(0,1e32),
               title='',fig='',plot=True):

    sph=fits.open(sptabfile)
    spt=sph['SPTAB'].data
    fbm=sph['FIBERMAP'].data
    w=( (spt['rv_adop'] >= rvrange[0])  
        & (spt['rv_adop'] <= rvrange[1])   
        & (spt['feh'] >= fehrange[0])
        & (spt['feh'] <= fehrange[1]) 
        & (spt['teff'] >= teffrange[0]) 
        & (spt['teff'] <= teffrange[1])
        & (spt['logg'] >= loggrange[0])
        & (spt['logg'] <= loggrange[1]) 
        & (spt['chisq_tot'] >= chisq_totrange[0])
        & (spt['chisq_tot'] <= chisq_totrange[1]) 
        & (spt['snr_med'] >= snr_medrange[0])
        & (spt['snr_med'] <= snr_medrange[1]) 
        & (fbm['target_ra'] >= rarange[0])      
        & (fbm['target_ra'] <= rarange[1])
        & (fbm['target_dec'] >= decrange[0]) 
        & (fbm['target_dec'] <= decrange[1]) 
        & (fbm['parallax'] >= parallaxrange[0])
        & (fbm['parallax'] <= parallaxrange[1]) 
        & (fbm['pmra'] >= pmrarange[0])
        & (fbm['pmra'] <= pmrarange[1]) 
        & (fbm['pmdec'] >= pmdecrange[0])
        & (fbm['pmdec'] <= pmdecrange[1]) )

    spt=spt[w]
    fbm=fbm[w]
    n=where(w)[0]

    if plot:
        plt.figure()
        plt.tight_layout()
        plt.ion()

        plt.subplot(3,2,1)
        plt.hist2d(spt['feh'],spt['alphafe'],120)
        plt.xlabel('[Fe/H]',labelpad=0)
        plt.ylabel('[a/Fe]')
        plt.title(sptabfile)

        plt.subplot(3,2,2)
        plt.hist(spt['teff'],bins=50)
        plt.xlabel('Teff',labelpad=0)
        plt.ylabel('N')
        plt.title(title)

        plt.subplot(3,2,3)
        plt.hist2d(spt['teff'],spt['logg'],120)
        plt.xlabel('Teff',labelpad=0)
        plt.ylabel('logg')
        plt.xlim([8000.,2000])
        plt.ylim([5.5,-0.5])

        plt.subplot(3,2,4)
        plt.hist2d(spt['teff'],spt['feh'],120)
        plt.xlabel('Teff',labelpad=0)
        plt.ylabel('[Fe/H]')
        plt.xlim([8000.,2000])
        plt.ylim([-5,1])


        plt.subplot(3,2,5)
        plt.hist(spt['rv_adop'],bins=50)
        plt.xlabel('RV',labelpad=0)
        plt.ylabel('N')


        plt.subplot(3,2,6)
        plt.hist(spt['feh'],bins=50)
        plt.xlabel('[Fe/H]',labelpad=0)
        plt.ylabel('N')
        me = median(spt['feh'])
        mm = mean(spt['feh'])
        ss = std(spt['feh'])
        plt.text(me-2*ss, 1, r'$\mu=$'+"{:5.2f}".format(mm) )
        plt.text(me+2*ss, 1, r'$\sigma=$'+"{:5.2f}".format(ss) )


        if fig != '': plt.savefig(fig)

        plt.show()

    return (spt,fbm)


#inspector
def inspect2(sptabfile,sym='.',rvrange=(-1e32,1e32),
               rarange=(0.,360.), decrange=(-90,90),  
               pmrarange=(-1e10,1e10), pmdecrange=(-1e10,1e10),
               parallaxrange=(-10000,10000),
               fehrange=(-100,100), teffrange=[0,100000],loggrange=[-100,100],
               chisq_totrange=(0.0,1e32),
               title='',fig='',plot=True):

    sph=fits.open(sptabfile)
    spt=sph['RVTAB'].data
    fbm=sph['FIBERMAP'].data
    w=( (spt['vrad'] >= rvrange[0])  
        & (spt['vrad'] <= rvrange[1])   
        & (spt['feh'] >= fehrange[0])
        & (spt['feh'] <= fehrange[1]) 
        & (spt['teff'] >= teffrange[0]) 
        & (spt['teff'] <= teffrange[1])
        & (spt['logg'] >= loggrange[0])
        & (spt['logg'] <= loggrange[1]) 
        & (spt['chisq_tot'] >= chisq_totrange[0])
        & (spt['chisq_tot'] <= chisq_totrange[1]) 
        & (fbm['target_ra'] >= rarange[0])      
        & (fbm['target_ra'] <= rarange[1])
        & (fbm['target_dec'] >= decrange[0]) 
        & (fbm['target_dec'] <= decrange[1]) 
        & (fbm['parallax'] >= parallaxrange[0])
        & (fbm['parallax'] <= parallaxrange[1]) 
        & (fbm['pmra'] >= pmrarange[0])
        & (fbm['pmra'] <= pmrarange[1]) 
        & (fbm['pmdec'] >= pmdecrange[0])
        & (fbm['pmdec'] <= pmdecrange[1]) )

    spt=spt[w]
    fbm=fbm[w]
    n=where(w)[0]

    if plot:
        plt.figure()
        plt.tight_layout()
        plt.ion()

        plt.subplot(3,2,1)
        plt.plot(spt['feh'],spt['alphafe'],sym)
        plt.xlabel('[Fe/H]',labelpad=0)
        plt.ylabel('[a/Fe]')
        plt.title(sptabfile)

        plt.subplot(3,2,2)
        plt.hist(spt['teff'],bins=50)
        plt.xlabel('Teff',labelpad=0)
        plt.ylabel('N')
        plt.title(title)

        plt.subplot(3,2,3)
        plt.plot(spt['teff'],spt['logg'],sym)
        plt.xlabel('Teff',labelpad=0)
        plt.ylabel('logg')
        plt.xlim([8000.,2000])
        plt.ylim([5.5,-0.5])

        plt.subplot(3,2,4)
        plt.plot(spt['teff'],spt['feh'],sym)
        plt.xlabel('Teff',labelpad=0)
        plt.ylabel('[Fe/H]')
        plt.xlim([8000.,2000])
        plt.ylim([-5,1])


        plt.subplot(3,2,5)
        plt.hist(spt['vrad'],bins=50)
        plt.xlabel('RV',labelpad=0)
        plt.ylabel('N')


        plt.subplot(3,2,6)
        plt.hist(spt['feh'],bins=50)
        plt.xlabel('[Fe/H]',labelpad=0)
        plt.ylabel('N')
        me = median(spt['feh'])
        mm = mean(spt['feh'])
        ss = std(spt['feh'])
        plt.text(me-2*ss, 1, r'$\mu=$'+"{:5.2f}".format(mm) )
        plt.text(me+2*ss, 1, r'$\sigma=$'+"{:5.2f}".format(ss) )


        if fig != '': plt.savefig(fig)

        plt.show()

    return (spt,fbm)


# pick-up metal-poor star candidates
def mpcandidates(sptabfile,minteff=4000.,maxteff=7000.,minfeh=-4.9,
    maxfeh=-4.0,minsnr_med=30.,maxchisq_tot=4.,sym='.'):

  s,m,h = read_tab(sptabfile)

  w = (s['teff'] > minteff) & (s['teff'] < maxteff) & (s['feh'] > minfeh) & (s['snr_med'] > minsnr_med) & (s['chisq_tot'] < maxchisq_tot) & (s['feh'] < maxfeh)

  plt.figure()
  plt.ion()
  plt.plot(s['teff'][w],s['logg'][w],sym)
  plt.xlabel('Teff (K)')
  plt.ylabel('logg (K)')
  plt.ylim([5.5,-1.0])
  plt.xlim([7000.,4000.])
  plt.show()

  ws = where(w)[0]
  print('  targetid/srcfile  Teff   logg [Fe/H] [a/Fe] snr_med chisq    ra        dec      Gmag')
  for i in ws:
    print('{:16}  {:7.2}  {:4.2}  {:5.2}  {:6.1}  {:6.1} {:4.2} {:10.8} {:10.8} {:4.3} \n {}'.format(s['targetid'][i],s['teff'][i],s['logg'][i],s['feh'][i], s['alphafe'][i],s['snr_med'][i],s['chisq_tot'][i],m['target_ra'][i],m['target_dec'][i],m['GAIA_PHOT_G_MEAN_MAG'][i],s['srcfile'][i]))

#peruse a spectrum
def peruse(targetid,sptabfile,sptabdir='./',subdirlevel=None):
	
  s, m, h = read_tab(sptabfile)
  w = (s['targetid'] == targetid)
  srcfile = s['srcfile'][w]
  
  print(srcfile)

  indir = sptabdir
  print(indir)
  if subdirlevel is not None: 
    for entry in range(subdirlevel):
      print(indir)
      indir = os.path.join(indir,'*')
  files = glob.glob(os.path.join(indir,'sptab_'+srcfile+'.fits'))
  print(indir,sptabfile)
  print(files)

  assert len(files) < 2,'more than one match found for the sptab '
 
  infile = files[0]
  s,f, h = read_tab(infile)
  w = where(s['targetid'] == targetid)[0]

  stop

  print('len(w)=',len(w))
  spmodfile = infile.replace('sptab','spmod')
  print(infile,spmodfile)
  bx,by, rx,ry, zx,zy, hm = read_spmod(spmodfile)
  
  plt.figure()
  plt.ion()
  plt.plot(bx,by['obs'][w,:],rx,ry['obs'][w,:],zx,zy['obs'][w,:])
  plt.plot(bx,by['fit'][w,:],rx,ry['fit'][w,:],zx,zy['fit'][w,:])
  plt.xlabel('Wavelength (A)')
  plt.ylabel('normalized flux')
  plt.title(targetid)
  plt.text(rx,mean(ry['obs'][w])/2,'Teff='+str(s['teff'][w]))
  plt.text(rx,mean(ry['obs'][w])/2.5,'logg='+str(s['logg'][w]))
  plt.text(rx,mean(ry['obs'][w])/3,'[Fe/H]='+str(s['feh'][w]))
  plt.text(rx,mean(ry['obs'][w])/3.5,'median(S/N)='+str(s['snr_med'][w]))
  plt.show()

#rv vs sp comparison
def rvspcomp(rvtabfile,sptabfile, clean=True):

  r, f1, h1 = read_tab(rvtabfile)
  s, f2, h2 = read_tab(sptabfile)

  rs, i1, i2 = intersect1d( r['targetid'], s['targetid'], return_indices=True)

  if clean:
    w=(r['teff'][i1] > 4000.) & (r['teff'][i1] < 7000.) & (r['feh'][i1] > -4.9) & (s['teff'][i2] > 4000.) & (s['teff'][i2] < 7000.) & (s['feh'][i2] > -4.9)  & (s['snr_med'][i2] > 10.) & (s['chisq_tot'][i2] < 4)  

    ww = where(w)[0]
    i1 = i1[ww]
    i2 = i2[ww]

  par = ['teff','logg','feh','alphafe']

  j = 1
  print(' SP - RV:  median  mean   std')
  for p in par:
    plt.subplot(2,2,j)
    plt.plot(r[p][i1],s[p][i2],',')
    plt.plot(r[p][i1],r[p][i1])
    plt.xlabel('rv '+p)
    plt.ylabel('sp '+p)
    print(p,median(s[p][i2]-r[p][i1]),mean(s[p][i2]-r[p][i1]),std(s[p][i2]-r[p][i1]))
    j = j + 1

  plt.show()

  return None

def apogeecomp(allstarfile,sptabfile,clean=True):
  
  allstar = fits.open(allstarfile)
  a = allstar[1].data
  s, f, h = read_tab(sptabfile)

  apo = SkyCoord(ra=a['ra']*units.degree, dec=a['dec']*units.degree)
  w =  (~isnan(apo.ra) & ~isnan(apo.dec)) #clean up single Nan
  apo = apo[w]
  desi = SkyCoord(ra=f['target_ra']*units.degree, dec=f['target_dec']*units.degree)
  
  max_sep = 1.0* units.arcsec
  idx, d2d, d3d = apo.match_to_catalog_sky(desi)
  i1 = d2d < max_sep
  i2 = idx[i1]
  ww = where(i1)[0]
  i1 = ww


  if clean:
    w=(a['teff'][i1] > 4000.) & (a['teff'][i1] < 7000.) & (a['m_h'][i1] > -4.9) & (a['aspcap_chi2'][i1] < 4) & (s['teff'][i2] > 4000.) & (s['teff'][i2] < 7000.) & (s['feh'][i2] > -4.9)  & (s['snr_med'][i2] > 10.) & (s['chisq_tot'][i2] < 4)  

    ww = where(w)[0]
    i1 = i1[ww]
    i2 = i2[ww]

  par = ['teff','logg','feh','alphafe']
  apopar = ['teff', 'logg', 'm_h','alpha_m']

  j = 1
  print(' SP - APOGEE:  median  mean   std')
  for p in par:
    print(len(i1),len(i2),len(a[apopar[j-1]]),len(s[p]))
    plt.subplot(2,2,j)
    plt.plot(a[apopar[j-1]][i1],s[p][i2],'.')
    plt.plot(a[apopar[j-1]][i1],a[apopar[j-1]][i1])
    plt.xlabel('APOGEE '+apopar[j-1])
    plt.ylabel('sp '+p)
    print(p,median(s[p][i2]-a[apopar[j-1]][i1]),mean(s[p][i2]-a[apopar[j-1]][i1]),std(s[p][i2]-a[apopar[j-1]][i1]))
    j = j + 1

  plt.show()

  return None

def ssppcomp(ssppfile,sptabfile,clean=True):
  
  allstar = fits.open(ssppfile)
  a = allstar[1].data
  s, f, h = read_tab(sptabfile)

  #cleanup missing decs
  w = a['dec'] > -90.
  a = a[w]

  apo = SkyCoord(ra=a['ra']*units.degree, dec=a['dec']*units.degree)
  desi = SkyCoord(ra=f['target_ra']*units.degree, dec=f['target_dec']*units.degree)
  
  max_sep = 1.0* units.arcsec
  idx, d2d, d3d = apo.match_to_catalog_sky(desi)
  i1 = d2d < max_sep
  i2 = idx[i1]
  ww = where(i1)[0]
  i1 = ww

  if clean:
    w=(a['teff_adop'][i1] > 4000.) & (a['teff_adop'][i1] < 7000.) & (a['feh_adop'][i1] > -4.9) & (s['teff'][i2] > 4000.) & (s['teff'][i2] < 7000.) & (s['feh'][i2] > -4.9)  & (s['snr_med'][i2] > 10.) & (s['chisq_tot'][i2] < 4)  

    ww = where(w)[0]
    i1 = i1[ww]
    i2 = i2[ww]

  par = ['teff','logg','feh','rv_adop']

  apopar = ['teff_adop', 'logg_adop', 'feh_adop','rv_adop']

  j = 1
  print(' SP - SEGUE/SSPP:  median  mean   std')
  for p in par:
    print(len(i1),len(i2),len(a[apopar[j-1]]),len(s[p]))
    plt.subplot(2,2,j)
    plt.plot(a[apopar[j-1]][i1],s[p][i2],',')
    plt.plot(a[apopar[j-1]][i1],a[apopar[j-1]][i1])
    plt.xlabel('SSPP '+apopar[j-1])
    plt.ylabel('sp '+p)
    print(p,median(s[p][i2]-a[apopar[j-1]][i1]),mean(s[p][i2]-a[apopar[j-1]][i1]),std(s[p][i2]-a[apopar[j-1]][i1]))
    j = j + 1

  plt.show()

  return None


def create_filters(modelfile,config='desi-n.yaml',libpath='.'):
  """Creates filter files from a dltfile (modelfile.dlt) and a config yaml file. 
  This is a very high level routine, meant to call synple's mkflt on each of the grids/bands
  involved in a particular configuration file.

  The complete process to produce the flt files is as follows:

   from synple import elements, polydelta, collectdelta

  a) get the first 99 chemical elements from elements()
   symbol, mass, sol = elements()

  b) setup spectral synthesis calculations for a model atmosphere and then 99 additional calculations for 0.2 dex changes in the abundances, e.g.
   polydelta('ksun.mod', (3000., 10000.), symbol)
   
  c) run those calculations by executing the 99 hyd*job files inside the hyd* folders

  d) collect the results in a 'dlt' file, e.g.
   collectdelta('ksun.mod', (3000.,10000.), symbol)

  e) smooth and resample the spectra in the dlt file, computing the response for each element. This is done from the derivatives, taking into account that lines overlap (for a given element we subtract the derivatives from all other). The flt files are obtained for each grid and band, and then bands are combined, e.g.
   create_filters('ksun.mod',config='desi-n.yaml',libpath='../grids')

  """

  from synple import elements, mkflt

  conf=load_conf(config,confdir=confdir)

  symbol, mass, sol = elements()

  grids = conf['grids']
  bands = conf['bands']

  for g in grids:
    for b in bands:
      gridfile = os.path.join(libpath,g+'-'+b+'.dat')
      x = lambda_synth(gridfile)
      hd = head_synth(gridfile)
      res = float(hd['RESOLUTION'])
      fwhm = 299792.458 / res # km/s
      print(g,b,gridfile,hd['RESOLUTION'],fwhm)
      tmpdir = g+'-'+b
      try:
        os.mkdir(tmpdir)
      except OSError:
        print( "cannot create folder %s " % (tmpdir) )
      mkflt(modelfile+'.dlt', x, fwhm=fwhm, outdir=tmpdir)

    
  
    for entry in symbol:
      file = entry+'.flt'
      print(g,file,g+'.'+file)
      f = open(g+'.'+file,'w')
      j = 0
      for b in bands:
        tmpdir = g+'-'+b
        data = loadtxt(os.path.join(tmpdir,file))
        if j == 0:
          res = data[:]
        else:
          res = concatenate((res,data))
        j = j + 1
      savetxt(f,res, fmt='%12.5e')
      f.close()

  return None

#transform equatorial to galactic coordinates
def radec2lb(ra,dec):

  sk = SkyCoord(ra=ra*units.degree, dec=dec*units.degree)
  gc = sk.transform_to('galactic')

  return(gc.l,gc.b)

#load a csv into a structured arrays (names are picked from the first row)
def loadcsv(csvfile):

  d=genfromtxt(csvfile,delimiter=',',names=True,skip_header=0)

  return(d)

#download a file over the net
#adapted from https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests/16696317#16696317
def download_file(url,local_filename=None, user=None, password=None):

    import requests
    from requests.auth import HTTPBasicAuth

    if local_filename is None: local_filename = url.split('/')[-1]
    if user is None or password is None:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    #if chunk: 
                    f.write(chunk)
    else:
        with requests.get(url, stream=True, auth = HTTPBasicAuth(user,password)) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    #if chunk: 
                    f.write(chunk)
    return local_filename

#Gaussian 
def gauss(x, B, A, mu,sigma):
  return B+A*exp(-(x-mu)**2/(2.*sigma**2))
    
#fit a Gaussian to data
def gaussfit(x, y, B=0.0, A=1.0, mu=0.0, sigma=1.0):
  p0 = [B, A, mu, sigma]
  coeff, var = curve_fit(gauss, x, y, p0=p0)
  
  return(coeff,var)
    
#cros-correlate two arrays of the same length and fit a Gaussian      
def xc(template,data,npoints=10,show=True):
  '''
  template is the reference array
  data is the data array
  npoints is the number of points used to fit a Gaussian to the xc function
  '''
  c = correlate(data,template,'same')
  lenc = len(c)
  hlenc = int(lenc/2)
  x = arange( lenc ) - hlenc
	
  #fitting a Gaussian
  if show:
   plt.plot(x,c,'*')
   plt.xlim([-npoints,npoints])
   lmin = min(c[hlenc-npoints:hlenc+npoints])
   lmax = max(c[hlenc-npoints:hlenc+npoints])
   plt.ylim([lmin*0.95,lmax*1.05])
  
  coeff, var = gaussfit(x[hlenc-int(npoints/2):hlenc+int(npoints/2)], 
               c[hlenc-int(npoints/2):hlenc+int(npoints/2)],
               B=min(c), A=max(c)-min(c), 
               mu=0.0, sigma=npoints/2.)
 
  if show:    
    plt.plot(x,gauss(x, *coeff))
    plt.show()
	
  return(coeff[2], sqrt(var[2,2]))

#cross correlate two spectra and fit a Gaussian to find the RV offset
def xcl(tlambda,template,dlambda,data,npoints=10,show=True):
  #tlabmbda and dlambda are the wavelength arrays for the spectra
  #in template and data, respectively
  #npoints is the number of data points considered in the Gaussian
  #fitting to the Cross-correlation function
	
  l0 = mean(tlambda)
  tv = (tlambda - l0) / l0 * clight
  dv = (dlambda - l0) / l0 * clight
  delta = median(diff(tv))
  nn = int( (max(tv) - min(tv) ) / delta ) + 1
  x = arange(nn) * delta + min(tv)
  rtemplate = interp(x, tv, template)
  rdata = interp(x, dv, data)
  offset, error = xc(rtemplate,rdata,npoints=npoints, show=show)
  print('offset is ',offset*delta, ' +/- ',error*delta, 'm/s')
  
  return(offset*delta, error*delta)
    
#check whether a tile has been observed
def check_tile(ra,dec):

    from getpass import getpass

    user = input("Please enter desi data username:\n")
    password = getpass("Please enter desi data password\n")
    download_file('https://data.desi.lbl.gov/desi/spectro/redux/daily/tiles-daily.csv',
      local_filename='tiles-tmp.csv', user=user,password=password)

    tiles = loadcsv('tiles-tmp.csv')

    w = (sqrt( (tiles['TILERA']*cos(tiles['TILEDEC']/180.*pi) - ra*cos(dec/180.*pi))**2 + (tiles['TILEDEC']-dec)**2 ) < 1.5 )
    
    if any(w): 
      print('the center of the following tiles falls within 1.5 deg from the target:\n',
            tiles.dtype.names,'\n ',tiles[where(w)[0]])
    else:
      print('no tiles have been processed whose center falls within 1.5 deg from the target:\n')
    
#create a series of testing folders to run a DESI data set over all possible permutations of the interpolation order

def mkindices(ndim,script='indices.sh',yaml='desi-s.yaml',sp='../../tiles/cumulative',spt='coadd',rv='../../rv_output',rvt='rvtab',c='desi-s2.yaml',l='../../../../grids'):

  import itertools
  o = open(script,'w')
  st=''
  for chiffre in range(ndim): st=st+str(chiffre+1)
  f=itertools.permutations(st,ndim)
  for entry in f: 
    s=''.join(list(entry))
    o.write('cp ~/piferre/config/'+yaml+' ~/piferre/config/'+c+' \n sustituye "indi:  1 2 3 4" "indi:  '+'  '.join(list(entry))+'" ~/piferre/config/'+c+' \n mkdir sp_s'+s+' \n cd sp_s'+s+' \n python3 ~/piferre/piferre.py -sp '+sp+' -spt '+spt+' -rv '+rv+' -rvt '+rvt+' -c '+c+' -l '+l+' \n cd .. \n')
  o.close()


#process a single pixel
def do(path, pixel, sdir='', truth=None, ncores=1, rvpath=None, 
libpath='.', sptype='spectra', rvtype='zbest', config='desi-n.yaml', only=[], cleanup=True):
  
  #get input data files
  #datafiles,zbestfiles  = finddatafiles(path,pixel,sdir,rvpath=rvpath) 

  print('do: path,rvpath,sdir,pixel=',path,rvpath,sdir,pixel)

  datafiles,zbestfiles  = getdata(sppath=os.path.join(path,sdir,pixel),
				rvpath=os.path.join(rvpath,sdir,pixel),
				sptype=sptype, rvtype=rvtype) 
  				  
  if (datafiles == None or zbestfiles == None or 
      len(datafiles) == 0 or len(zbestfiles) == 0): return None

  #identify data source
  datafile=datafiles[0]
  hdu=fits.open(datafile)
  enames=extnames(hdu)
  if 'FIBERMAP' in enames: 
    source='desi'
  else:
    source='boss'

  #gather config. info
  conf=load_conf(config,confdir=confdir)
  #set the set of grids to be used
  grids=conf['grids']
  #print('grids=',grids)
  bands=conf['bands']
  grid_bands=conf['grid_bands']
  if conf['seconds_per_spectrum']: 
    seconds_per_spectrum=conf['seconds_per_spectrum']
  else:
    seconds_per_spectrum=10.


  #loop over possible multiple data files in the same pixel
  for ifi in range(len(datafiles)):

    datafile=datafiles[ifi]
    zbestfile=zbestfiles[ifi]
    fileroot=datafile.split('.')[-2].split('/')[-1]
    print('datafile=',datafile)
    print('fileroot=',fileroot)

    #get redshifts
    if (zbestfile.find('best') > -1) or (zbestfile.find('redrock') > -1):
      z=read_zbest(zbestfile)
    else:
      #Koposov pipeline
      z=read_k(zbestfile)
  
    #read primary header and  
    #find out if there is FIBERMAP extension
    #identify MWS targets
    hdu=fits.open(datafile)
    enames=extnames(hdu)
    pheader=hdu['PRIMARY'].header
    #print('datafile='+datafile)
    #print('extensions=',enames)

    if source == 'desi': #DESI data
      fibermap=hdu['FIBERMAP']
      if 'SCORES' in enames: scores=hdu['SCORES']
      targetid=fibermap.data['TARGETID']
      if 'FIBER' in fibermap.data.names:
        fiber=fibermap.data['FIBER']
      else:
        fiber=zeros(len(targetid),dtype=int) 
      if 'RA_TARGET' in fibermap.data.names: 
        ra=fibermap.data['RA_TARGET']
      else:
        if 'TARGET_RA' in fibermap.data.names:
          ra=fibermap.data['TARGET_RA']
        else:
          ra=-9999.*ones(len(targetid))
      if 'DEC_TARGET' in fibermap.data.names:
        dec=fibermap.data['DEC_TARGET']
      else:
        if 'TARGET_DEC' in fibermap.data.names:
          dec=fibermap.data['TARGET_DEC']
        else:
          dec=-9999.*ones(len(targetid))
      if 'MAG' in fibermap.data.names: 
        mag=fibermap.data['MAG']
      else:
          mag=[-9999.*ones(5)]
          for kk in range(len(targetid)-1): mag.append(-9999.*ones(5))
      nspec=ra.size
      

    else:  #SDSS/BOSS data

      plate=pheader['PLATEID']
      mjd=pheader['MJD']
      #fibermap=hdu['PLUGMAP']
      fibermap=hdu[5]
      fiberid=fibermap.data['fiberid']
      fiber=array(fiberid,dtype=int)
      ra=fibermap.data['RA']
      dec=fibermap.data['DEC']
      #mag=zeros((ra.size,5)) # used zeros for LAMOST fibermap.data['MAG']
      mag=fibermap.data['MAG']
      nspec=ra.size
      targetid=[]
      for i in range(nspec): 
        targetid.append(str(plate)+'-'+str(mjd)+'-'+str(fiberid[i]))

      targetid=array(targetid)


    #identify targets to process based on redshift: 0.00<=|z|<0.01
    process_target = zeros(nspec, dtype=bool)
    for i in range(nspec):
      if z.get(targetid[i],-1) != -1:
        if (abs(z[targetid[i]][0]) < 0.01) & (abs(z[targetid[i]][0]) >= 0.): process_target[i]= True

    #if the array 'only' is not emtpy, only the fibers indicated will be processed
    #only can have strings and integers: 
    #   strings are matched against targetid
    #   integers are matched against fiber
    if len(only) > 0:
      process_target = zeros(nspec, dtype=bool)
      for entry in only:
        try: 
          entry = int(entry)
          w = (fiber == entry)
        except ValueError:
          w = (targetid == entry)
        process_target[w] = True
        

    
    #skip the rest of the code if there are no targets 
    #or the wavelengths for one band are not present
    if (process_target.nonzero()[0].size == 0): 
      print('Warning: no targets selected -- skipping datafile ', datafile)
      continue
    
    complete = 1
    for i in range(len(bands)):
      if bands[i].upper()+'_WAVELENGTH' not in enames: 
        print('Warning: missing ',bands[i].upper()+'_WAVELENGTH',' in  enames -- skipping datafile ', datafile)
        complete = 0
    if (complete == 0): continue

    #set ids array (with targetids) and par dictionary for vrd/ipf file
    ids=[]
    par={}

    #truth (optional, for simulations)
    #if (len(sys.argv) == 3):
    npass=0 #count targets that pass the filter (process_target)
    if truth is not None:
      (true_feh,true_teff,true_logg,true_rmag,true_z)=truth
      for k in range(nspec):
        if process_target[k]:
          npass=npass+1
          if source == 'desi': id = str(targetid[k]) + '-' + str(fiber[k])
          ids.append(id)
          par[id]=[true_feh[targetid[k]],true_teff[targetid[k]],
                         true_logg[targetid[k]],true_rmag[targetid[k]],
			 true_z[targetid[k]],z[targetid[k]][0],z[targetid[k]][1],
                         ra[k],dec[k]]
          #stop
    else:
      for k in range(nspec):
        if process_target[k]:
          npass=npass+1
          if source == 'desi': id = str(targetid[k]) + '_' + str(fiber[k])
          ids.append(id)
          #we patch the redshift here to handle missing redshifts for comm. data from Sergey
          #z[targetid[k]]=0.0
          par[id]=[0.0,0.0,0.0,mag[k][2],0.0,z[targetid[k]][0],z[targetid[k]][1],ra[k],dec[k]]
          #stop        

    #collect data for each band
    for j in range(len(bands)):

      if len(grid_bands) == 0:
        gridfile=grids[0]+'.dat'
      else:
        if len(grid_bands) == 1:
          gridfile=grids[0]+'-'+grid_bands[0]+'.dat'
        elif len(grid_bands) == len(bands):
          gridfile=grids[0]+'-'+bands[j]+'.dat'
        else:
          print('do: error -- the array grid_bands must have 0, 1 or the same length as bands')
          return None   

      #read grid wavelength array
      x1=lambda_synth(os.path.join(libpath,gridfile))
      if len(x1) == len(bands): x1=x1[j]

      #read DESI data, select targets, and resample 

      (x,y,ivar,r)=read_spec(datafile,bands[j])
      ey=sqrt(divide(1.,ivar,where=(ivar > 0.)))
      ey[where(ivar == 0.)]=max(y)*1e3

      #plt.ion()
      #plt.plot(x,y[0])
      #plt.show()
      #plt.plot(x,y[0])
      #plt.show()

      nspec, freq = y.shape
      if j == 0:
        print('nspec=',nspec)    
        print('n(process_target)=',process_target.nonzero()[0].size)
      y2=zeros((npass,len(x1)))
      ey2=zeros((npass,len(x1)))
      k=0
      #print('nspec,len(z),npass,len(x1)=',nspec,len(z),npass,len(x1))
      for i in range(nspec):
        if process_target[i]:
          y2[k,:]=interp(x1,x*(1.-z[targetid[i]][0]),y[i,:])
          ey2[k,:]=interp(x1,x*(1-z[targetid[i]][0]),ey[i,:])
          k=k+1

      if (j==0):
        xx=x1
        yy=y2
        eyy=ey2
      else:
        xx=concatenate((xx,x1))
        yy=concatenate((yy,y2),axis=1)
        eyy=concatenate((eyy,ey2),axis=1)

      savetxt(os.path.join(sdir,pixel,fileroot)+'-'+bands[j]+'.wav',x1,fmt='%14.5e')

    savetxt(os.path.join(sdir,pixel,fileroot)+'.wav',xx,fmt='%14.5e')
    fmp = tbl.Table(fibermap.data) [process_target]
    hdu0 = fits.BinTableHDU(fmp)
    hdu0.writeto(os.path.join(sdir,pixel,fileroot)+'.fmp.fits')
    if 'SCORES' in enames: 
      scr = tbl.Table(scores.data) [process_target]
      hdu0 = fits.BinTableHDU(scr)
      hdu0.writeto(os.path.join(sdir,pixel,fileroot)+'.scr.fits')

    write_ferre_input(fileroot,ids,par,yy,eyy,path=os.path.join(sdir,pixel))
     
    minutes= 10. + npass*seconds_per_spectrum/60.*32./ncores

    #write slurm script
    write_slurm(fileroot,path=os.path.join(sdir,pixel),
            ngrids=len(grids),ncores=ncores, minutes=minutes, config=config, cleanup=cleanup)


    #loop over all grids
    mknml(conf,fileroot,libpath=libpath,path=os.path.join(sdir,pixel))

    #run ferre
    #ferrerun(path=os.path.join(sdir,pixel))

    #opfmerge(pixel,path=os.path.join(sdir,pixel))


  return None


def main(args):

  parser = argparse.ArgumentParser(description='prepare a data set for processing with FERRE')

  parser.add_argument('-sp','--sppath',
                      type=str,
                      help='path to the input spectra dir tree (must contain directories)',
                      default=None)

  parser.add_argument('-rv','--rvpath',
                      type=str,
                      help='path to the RV input data',
                      default=None)

  parser.add_argument('-l','--libpath',
                      type=str,
                      help='path to the libraries, if not in the current dir',
                      default='.')

  parser.add_argument('-spt','--sptype',
                      type=str,
                      help='type of data (spectra, coadd, spPlate)',
                      default='spectra')

  parser.add_argument('-rvt','--rvtype',
                      type=str,
                      help='type of RV data (zbest, rvtab_spectra, rvtab_coadd, spZbest)',
                      default='zbest')

  parser.add_argument('-c','--config',
                      type=str,
                      help='yaml configuration file for FERRE runs',
                      default='desi-n.yaml')

  parser.add_argument('-n','--ncores',
                      type=int,
                      help='number of cores per slurm job',
                      default=32)
                      
  parser.add_argument('-t','--truthfile',
                      type=str,
                      help='truth file for DESI simulations',
                      default=None)

  parser.add_argument('-o','--only',
                      type=list,
                      help='list of target ids or fiber numbers to process',
                      default=[])

  parser.add_argument('-x','--no-cleanup', dest='cleanup', action='store_false')
  parser.set_defaults(cleanup=True)


  args = parser.parse_args()

  sppath=args.sppath
  rvpath=args.rvpath
  if rvpath is None: rvpath=sppath

  libpath=args.libpath

  sptype=args.sptype
  rvtype=args.rvtype

  config=args.config
  ncores=args.ncores

  truthfile=args.truthfile
  if (truthfile is not None):  truthtuple=read_truth(truthfile)
  else: truthtuple=None

  only=args.only

  cleanup=args.cleanup

  pixels=getpixels(sppath)
  
  dopars = []
  for entry in pixels:
    head, pixel = os.path.split(entry)
    #print('head/pixel=',head,pixel)
    sdir=''
    #print('sppath=',sppath)
    #print('rvpath=',rvpath)
    if head != sppath:
      head, sdir = os.path.split(head)
      if not os.path.exists(sdir): os.mkdir(sdir)
    if sdir != '': 
      if not os.path.exists(sdir):os.mkdir(sdir)
    if not os.path.exists(os.path.join(sdir,pixel)): 
      os.mkdir(os.path.join(sdir,pixel))

    pararr = [sppath,pixel,sdir,truthtuple,ncores, 
       rvpath, libpath, sptype, rvtype, config, only, cleanup]

    #do(sppath,pixel,sdir=sdir,truth=truthtuple, 
    #   rvpath=rvpath, libpath=libpath, 
    #   sptype=sptype, rvtype=rvtype,
    #   ncores=ncores,  config=config)

    #run(pixel,path=os.path.join(sdir,pixel))
    
    print('cleanup=',cleanup)

    dopars.append(pararr)

  #ncores = cpu_count()
  pool = Pool(32)
  results = pool.starmap(do,dopars)

  
if __name__ == "__main__":
  main(sys.argv[1:])
