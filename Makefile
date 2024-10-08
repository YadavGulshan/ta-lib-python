.PHONY: build

build:
	python3 -m pip install --upgrade build
	python3 -m build

install:
	python3 -m pip install .

debug: cython install generate

_func: tools/generate_func.py
	python3 tools/generate_func.py > talib/_func.pxi.tmp && \
	mv talib/_func.pxi.tmp talib/_func.pxi || \
	(rm -f talib/_func.pxi.tmp; false)


_stream: tools/generate_stream.py
	python3 tools/generate_stream.py > talib/_stream.pxi.tmp && \
	mv talib/_stream.pxi.tmp talib/_stream.pxi || \
	(rm -f talib/_stream.pxi.tmp; false)

generate: _func _stream

cython:
	cython --directive emit_code_comments=False talib/_ta_lib.pyx

clean:
	rm -rf build talib/_ta_lib.so talib/*.pyc

perf:
	python3 tools/perf_talib.py

test: build
	LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH} pytest

sdist:
	python3 setup.py sdist --formats=gztar,zip
