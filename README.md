# hduplooy/gomatrix

## Basic implementation of matrix routines in golang

### API


func CloneVector(val []float64) []float64
    CloneVector just makes a copy of a slice

func Cols(m Matrix) int
    Cols is an utility function to return the number of columns within the
    Matrix

func Cramer(m Matrix, r []float64) ([]float64, error)
    Cramer implements Cramer's rule of finding the solution to a system of
    linear equations Other faster solutions will be added over time

func Determinant(m Matrix) (float64, error)
    Determinant - normal matrix determinant The determinant and an error is
    returned This is the currently the slow recursive solution (will be
    amended later)

func Equal(m1, m2 Matrix) bool
    Equal will return whether two matrices are the same This is not too
    useful seeing that one has rounding errors Will add another one to
    measure equality within certain margins later

func GetCol(m Matrix, n int) []float64
    GetCol will extract the row n from the matrix and return it as a slice

func GetRow(m Matrix, n int) []float64
    GetRow will extract the row n from the matrix and return it as a slice

func IsDiagonal(m Matrix) bool
    IsDiagonal check whether only the diagonal has values

func IsIdentity(m Matrix) bool
    IsIdentity check whether the matrix is an identity matrix

func IsLowerTriangular(m Matrix) bool
    IsLowerTriangular check whether the left lower corner only has values

func IsOrthogonal(m Matrix) bool
    IsOrthogonal check whether the transpose of a matrix equals the inverse

func IsSingleEntry(m Matrix) bool
    IsSingleEntry check if only one cell has a value

func IsSquare(m Matrix) bool
    IsSquare returns whether the matrix has the same number of rows as
    columns

func IsSymmetric(m Matrix) bool
    IsSymmetric check whether the matrix equals its transpose

func IsUpperTriangular(m Matrix) bool
    IsUpperTriangular check whether the right upper corner only has values

func Print(w io.Writer, m Matrix, wdth, aft int)
    Print will print the matrix to the w stream with wdth size for each cell
    and aft decimals

func Rows(m Matrix) int
    Rows is an utility function to return the number of rows within the
    Matrix

func SetCol(m *Matrix, n int, val []float64)
    SetCol will alter m by setting the n'th col to val Nothing is returned,
    m is changed The number of rows of the matrix must be the same as the
    length of val

func SetRow(m *Matrix, n int, val []float64)
    SetRow will alter m by setting the n'th row to val Nothing is returned,
    m is changed The number of columns of the matrix must be the same as the
    length of val

func ToString(m Matrix, wdth, aft int) []string
    ToString will generate a string representation of the matrix with each
    cell of size wdth and aft decimals The lines are returned as a slice

func Trace(m Matrix) float64
    Trace returns the sum of the diagonal values

TYPES

type Matrix [][]float64
    A Matrix is just a 2 dimensional slice of type float64

func Add(m ...Matrix) Matrix
    Add does a cell wise addition of the matrices returning the result If
    the dimensions are not the same a nil Matrix is returned

func AddScalar(m Matrix, n float64) Matrix
    AddScalar will add the value n to each element within the matrix and
    return the result

func Clone(m Matrix) Matrix
    Clone will make an exact copy of a Matrix

func ColVectorToMatrix(v []float64) Matrix
    ColVectorToMatrix will take the slice v an return it as a Matrix with
    only one column

func CreateMatrix(rows, cols int) Matrix
    CreateMatrix will create and init the 2 dimensional slice

func DivDot(m1, m2 Matrix) Matrix
    DivDot will do a cell wise division between two matrices The rows and
    columns must be the same

func DivScalar(m Matrix, n float64) Matrix
    DivScalar will divide each element by the value n

func DotFunc2(m1, m2 Matrix, f func(float64, float64) float64) Matrix
    DotFunc2 will take cells from the two matrices and apply f to it
    generating a new matrix that is returned m1 and m2 must be the same
    dimensions

func DotFuncN(f func(v ...float64) float64, m ...Matrix) Matrix
    DotFuncN will take cells from all the matrices and apply f to it
    generating a new matrix that is returned It is necessary to make sure
    that the number of variables that f expect is the same as the number of
    matrices provided all matrices must be the same dimensions This is
    similar to the map function used by various lisp and other languages

func Exclude(m Matrix, row, col int) Matrix
    Exclude will return a new matrix with the specific row and column
    excluded from it

func IdentityMatrix(n int) Matrix
    IndentiyMatrix will generate an identity matrix of dimension nxn

func Inverse(m Matrix) (Matrix, error)
    Inverse will calculate the inverse of a matrix if it is square and the
    determinant is not zero

func InvertHorizontal(m Matrix) Matrix
    InvertHorizontal will invert each row of m

func InvertVertical(m Matrix) Matrix
    InvertVertical will invert each column of m

func Mult(m ...Matrix) Matrix
    Mult will multiply the provided matrices with each other (using Mult2)

func Mult2(m1 Matrix, m2 Matrix) Matrix
    Mult2 will multiply two matrices This is the standard slow method, the
    faster alternatives will be added later Note that the columns of m1 must
    be the same as the rows of m2

func MultDot(m1, m2 Matrix) Matrix
    MultDot will do a cell wise multiplication between two matrices The rows
    and columns must be the same

func MultScalar(m1 Matrix, val float64) Matrix
    MultScalar multiplies the value val to each cell of m1 and return the
    result

func Pow(m Matrix, n int) Matrix
    Pow will return the integer power of a matrix if it is square

func PowDot(m1, m2 Matrix) Matrix
    PowDot will calc the exponent of the cell in m1 to the cell in m2
    returning the result The rows and columns must be the same

func PowScalar(m Matrix, n float64) Matrix
    PowScalar will take the exponent of each cell to the value n returning
    the new matrix

func RowVectorToMatrix(v []float64) Matrix
    RowVectorToMatrix will take the slice v and return it as a Matrix with
    only one row

func ScalarFunc(m Matrix, f func(float64) float64) Matrix
    MultFunc will apply a func to each cell return the new matrix The func
    in this instance has only one value

func Sub(m ...Matrix) Matrix
    Add does a cell wise subtraction of the matrices returning the result If
    the dimensions are not the same a nil Matrix is returned

func SubMatrix(m Matrix, strow, stcol, rows, cols int) Matrix
    SubMatrix will extract a piece of a matrix strow - start row stcol -
    start col rows - number of rows to take cols - number of cols to take

func SubScalar(m Matrix, n float64) Matrix
    SubScalar will subtract the value n from each cell returning the result

func SwapCols(m Matrix, col1, col2 int) Matrix
    SwapCols will swap columns col1 and col2 and return the result

func SwapRows(m Matrix, row1, row2 int) Matrix
    SwapRows will swap rows row1 and row2 and return the result

func Transpose(m Matrix) Matrix
    Transpose will do a normal matrix transpose and return it

