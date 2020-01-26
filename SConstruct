# -*-Python-*-
import os

Clean('build', 'build')

VariantDir('build/normal', 'src', duplicate=0)

env = Environment()
env['CPPFLAGS'] = ["-O2", "-g", "--std=c++11"]
env['LIBS'] = ['m', 'OpenCL']
env['DIR'] = 'build/normal'

env.SConscript('${DIR}/SConscript',
               exports='env')
