.PHONY: clean all
all: comm-time.mpi \
	 pi-calc.mpi

%.mpi: %.cc
	mpic++ -Wall -Wpedantic -lgmp $< -o $@

clean:
	rm -f *.mpi
