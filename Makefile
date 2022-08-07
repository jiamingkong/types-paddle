#
# Makefile for types-paddle
#

IGNORE_PEP=E203,E221,E241,E272,E501,F811
.PHONY: all
all : clean lint

.PHONY: clean
clean:
	rm -fr dist/* .pytype

.PHONY: install
install:
	pip3 install -r requirements.txt
	pip3 install -r requirements-dev.txt

.PHONY: dist
dist:
	pip uninstall -y types-paddle
	rm -rf dist
	python3 setup.py sdist bdist_wheel
	pip install dist/types_paddle-0.0.2-py3-none-any.whl

.PHONY: publish
publish:
	PATH=~/.local/bin:${PATH} twine upload dist/*

.PHONY: version
version:
	@newVersion=$$(awk -F. '{print $$1"."$$2"."$$3+1}' < VERSION) \
		&& echo $${newVersion} > VERSION \
		&& git add VERSION \
		&& git commit -m "🔥 update version to $${newVersion}" > /dev/null \
		&& git tag "v$${newVersion}" \
		&& echo "Bumped version to $${newVersion}"

.PHONY: doc
doc:
	mkdocs serve
