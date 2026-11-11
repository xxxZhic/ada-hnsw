PYTHON ?= python

.PHONY: build test clean
build:
	$(PYTHON) setup.py build_ext --inplace
test: build
	$(PYTHON) -m unittest discover -s ../tests -v
clean:
	rm -rf build *.egg-info
	rm -f ada_hnswlib*.so
