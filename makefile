# INSTALL ##############################

install :
	cd GCMex && matlab -nodisplay -r "try compile_gc; catch; end; quit"
