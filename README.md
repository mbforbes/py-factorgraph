# py-factorgraph

[![Build Status](https://travis-ci.org/mbforbes/py-factorgraph.svg?branch=master)](https://travis-ci.org/mbforbes/py-factorgraph)
[![Coverage Status](https://coveralls.io/repos/github/mbforbes/py-factorgraph/badge.svg?branch=master)](https://coveralls.io/github/mbforbes/py-factorgraph?branch=master)
[![license MIT](http://b.repl.ca/v1/license-MIT-brightgreen.png)](https://github.com/mbforbes/py-factorgraph/blob/master/LICENSE.txt)

Factor graphs and loopy belief propagation implemented in Python.

## installation

```bash
pip install factorgraph
```

## thanks

- to Matthew R. Gormley and Jason Eisner for the [Structured Belief Propagation
  for NLP Tutorial](https://www.cs.cmu.edu/~mgormley/bp-tutorial/), which was
  extremely helpful for me in learning about factor graphs and understanding
  the sum product algorithm.

- to Ryan Lester for [pyfac](https://github.com/rdlester/pyfac), whose tests I
  used directly to test my implementation

## TODO

-	[x] graph
-	[x] bp/lbp
-	[x] repo structure
-	[x] tests
-	[x] viz (another repo)
-	[x] cleanup (pep8, requirements.txt, use logging lib, etc.)
-	[x] CI
-   [ ] coveralls
-	[ ] Readme
    -   [ ] figure out good structure for this. also don't forget:
    -   [ ] BP tutorial link
    -   [ ] note pyfac for inspiration and test help
    -   [ ] installation
    -   [ ] examples
    -   [ ] viz pic + link
    -	[ ] future work: API cleanup, timing and optimization, optional logging
    -   [ ] projects using this

