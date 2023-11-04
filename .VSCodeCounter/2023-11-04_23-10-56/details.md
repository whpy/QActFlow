# Details

Date : 2023-11-04 23:10:56

Directory /home/hwu/Projects/QActFlow

Total : 54 files,  5357 codes, 993 comments, 934 blanks, all 7284 lines

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [README.md](/README.md) | Markdown | 3 | 0 | 1 | 4 |
| [src/Basic/Field.cpp](/src/Basic/Field.cpp) | C++ | 9 | 0 | 3 | 12 |
| [src/Basic/Field.h](/src/Basic/Field.h) | C++ | 13 | 0 | 3 | 16 |
| [src/Basic/FldOp.cu](/src/Basic/FldOp.cu) | CUDA C++ | 171 | 44 | 25 | 240 |
| [src/Basic/FldOp.cuh](/src/Basic/FldOp.cuh) | CUDA C++ | 73 | 18 | 25 | 116 |
| [src/Basic/FldOp.h](/src/Basic/FldOp.h) | C++ | 67 | 16 | 23 | 106 |
| [src/Basic/FldOp.hpp](/src/Basic/FldOp.hpp) | C++ | 165 | 39 | 26 | 230 |
| [src/Basic/Mesh.cpp](/src/Basic/Mesh.cpp) | C++ | 51 | 7 | 6 | 64 |
| [src/Basic/Mesh.h](/src/Basic/Mesh.h) | C++ | 24 | 6 | 7 | 37 |
| [src/Basic/QActFlow.h](/src/Basic/QActFlow.h) | C++ | 16 | 0 | 3 | 19 |
| [src/Basic/QActFlowDef.hpp](/src/Basic/QActFlowDef.hpp) | C++ | 7 | 0 | 4 | 11 |
| [src/Basic/cuComplexBinOp.hpp](/src/Basic/cuComplexBinOp.hpp) | C++ | 145 | 6 | 33 | 184 |
| [src/Basic/cudaErr.h](/src/Basic/cudaErr.h) | C++ | 55 | 3 | 22 | 80 |
| [src/BasicUtils/UtilFuncs.hpp](/src/BasicUtils/UtilFuncs.hpp) | C++ | 99 | 0 | 9 | 108 |
| [src/Qsolver/Qsolver.cu](/src/Qsolver/Qsolver.cu) | CUDA C++ | 165 | 48 | 36 | 249 |
| [src/Qsolver/Qsolver.cuh](/src/Qsolver/Qsolver.cuh) | CUDA C++ | 15 | 0 | 11 | 26 |
| [src/Qsolver/structure_visualize.m](/src/Qsolver/structure_visualize.m) | Objective-C | 118 | 0 | 7 | 125 |
| [src/Stream/Streamfunc.cu](/src/Stream/Streamfunc.cu) | CUDA C++ | 235 | 166 | 45 | 446 |
| [src/Stream/Streamfunc.cuh](/src/Stream/Streamfunc.cuh) | CUDA C++ | 34 | 0 | 18 | 52 |
| [src/Stream/StreamfuncModified.cu](/src/Stream/StreamfuncModified.cu) | CUDA C++ | 236 | 53 | 37 | 326 |
| [src/Stream/StreamfuncModified.cuh](/src/Stream/StreamfuncModified.cuh) | CUDA C++ | 28 | 8 | 17 | 53 |
| [src/TimeIntegration/RK4.cu](/src/TimeIntegration/RK4.cu) | CUDA C++ | 102 | 27 | 6 | 135 |
| [src/TimeIntegration/RK4.cuh](/src/TimeIntegration/RK4.cuh) | CUDA C++ | 26 | 1 | 10 | 37 |
| [src/UnitTest/Basic/BFwd.cu](/src/UnitTest/Basic/BFwd.cu) | CUDA C++ | 70 | 0 | 8 | 78 |
| [src/UnitTest/Basic/BasicOp.cu](/src/UnitTest/Basic/BasicOp.cu) | CUDA C++ | 63 | 10 | 9 | 82 |
| [src/UnitTest/Basic/structure_visualize.m](/src/UnitTest/Basic/structure_visualize.m) | Objective-C | 101 | 0 | 4 | 105 |
| [src/UnitTest/BasicUtils/readcsv.cu](/src/UnitTest/BasicUtils/readcsv.cu) | CUDA C++ | 28 | 0 | 2 | 30 |
| [src/UnitTest/ModifiedStream/nonlterms.cu](/src/UnitTest/ModifiedStream/nonlterms.cu) | CUDA C++ | 376 | 45 | 47 | 468 |
| [src/UnitTest/Qsolver.cu](/src/UnitTest/Qsolver.cu) | CUDA C++ | 165 | 48 | 36 | 249 |
| [src/UnitTest/Qsolver.cuh](/src/UnitTest/Qsolver.cuh) | CUDA C++ | 15 | 0 | 11 | 26 |
| [src/UnitTest/Qsolver/Qsolver.cu](/src/UnitTest/Qsolver/Qsolver.cu) | CUDA C++ | 68 | 3 | 11 | 82 |
| [src/UnitTest/Qsolver/Qsolver.cuh](/src/UnitTest/Qsolver/Qsolver.cuh) | CUDA C++ | 15 | 0 | 11 | 26 |
| [src/UnitTest/Qsolver/a_0_025|Re_0_1/Qsolver.cu](/src/UnitTest/Qsolver/a_0_025%7CRe_0_1/Qsolver.cu) | CUDA C++ | 177 | 73 | 34 | 284 |
| [src/UnitTest/Qsolver/a_0_025|Re_0_1/Qsolver.cuh](/src/UnitTest/Qsolver/a_0_025%7CRe_0_1/Qsolver.cuh) | CUDA C++ | 19 | 1 | 11 | 31 |
| [src/UnitTest/Qsolver/a_0_025|Re_0_1/structure_visualize.m](/src/UnitTest/Qsolver/a_0_025%7CRe_0_1/structure_visualize.m) | Objective-C | 27 | 0 | 8 | 35 |
| [src/UnitTest/Qsolver/console_a_0_025/Qsolver.cu](/src/UnitTest/Qsolver/console_a_0_025/Qsolver.cu) | CUDA C++ | 169 | 48 | 36 | 253 |
| [src/UnitTest/Qsolver/console_a_0_025/Qsolver.cuh](/src/UnitTest/Qsolver/console_a_0_025/Qsolver.cuh) | CUDA C++ | 15 | 0 | 11 | 26 |
| [src/UnitTest/Qsolver/console_a_0_025/structure_visualize.m](/src/UnitTest/Qsolver/console_a_0_025/structure_visualize.m) | Objective-C | 27 | 0 | 8 | 35 |
| [src/UnitTest/Qsolver/console_a_0_20/Qsolver.cu](/src/UnitTest/Qsolver/console_a_0_20/Qsolver.cu) | CUDA C++ | 170 | 48 | 36 | 254 |
| [src/UnitTest/Qsolver/console_a_0_20/Qsolver.cuh](/src/UnitTest/Qsolver/console_a_0_20/Qsolver.cuh) | CUDA C++ | 15 | 0 | 11 | 26 |
| [src/UnitTest/Qsolver/console_a_0_20/structure_visualize.m](/src/UnitTest/Qsolver/console_a_0_20/structure_visualize.m) | Objective-C | 27 | 0 | 8 | 35 |
| [src/UnitTest/Qsolver/restart/Qsolver.cu](/src/UnitTest/Qsolver/restart/Qsolver.cu) | CUDA C++ | 181 | 57 | 38 | 276 |
| [src/UnitTest/Qsolver/restart/Qsolver.cuh](/src/UnitTest/Qsolver/restart/Qsolver.cuh) | CUDA C++ | 19 | 1 | 11 | 31 |
| [src/UnitTest/Qsolver/restart/structure_visualize.m](/src/UnitTest/Qsolver/restart/structure_visualize.m) | Objective-C | 27 | 0 | 8 | 35 |
| [src/UnitTest/Qsolver/structure_visualize.m](/src/UnitTest/Qsolver/structure_visualize.m) | Objective-C | 16 | 0 | 6 | 22 |
| [src/UnitTest/Streamfunc/crossterm.cu](/src/UnitTest/Streamfunc/crossterm.cu) | CUDA C++ | 335 | 75 | 43 | 453 |
| [src/UnitTest/Streamfunc/make.sh](/src/UnitTest/Streamfunc/make.sh) | Shell Script | 2 | 1 | 3 | 6 |
| [src/UnitTest/Streamfunc/nonlterms.cu](/src/UnitTest/Streamfunc/nonlterms.cu) | CUDA C++ | 376 | 45 | 47 | 468 |
| [src/UnitTest/Streamfunc/nonlterms2.cu](/src/UnitTest/Streamfunc/nonlterms2.cu) | CUDA C++ | 379 | 45 | 47 | 471 |
| [src/UnitTest/Streamfunc/structure_visualize.m](/src/UnitTest/Streamfunc/structure_visualize.m) | Objective-C | 128 | 0 | 6 | 134 |
| [src/UnitTest/TimeIntegration/burgers1D.cu](/src/UnitTest/TimeIntegration/burgers1D.cu) | CUDA C++ | 148 | 26 | 19 | 193 |
| [src/UnitTest/TimeIntegration/nonlinear1.cu](/src/UnitTest/TimeIntegration/nonlinear1.cu) | CUDA C++ | 129 | 25 | 19 | 173 |
| [src/UnitTest/TimeIntegration/structure_visualize.m](/src/UnitTest/TimeIntegration/structure_visualize.m) | Objective-C | 112 | 0 | 4 | 116 |
| [structure_visualize.m](/structure_visualize.m) | Objective-C | 101 | 0 | 4 | 105 |

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)