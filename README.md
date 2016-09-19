# hduplooy/gomatrix

## Basic implementation of matrix routines in golang

### API


**func CloneVector(val []float64) []float64**    
&nbsp;&nbsp;&nbsp;&nbsp;CloneVector just makes a copy of a slice

**func Cols(m Matrix) int**    
&nbsp;&nbsp;&nbsp;&nbsp;Cols is an utility function to return the number of columns within the
&nbsp;&nbsp;&nbsp;&nbsp;Matrix

**func Cramer(m Matrix, r []float64) ([]float64, error)**    
&nbsp;&nbsp;&nbsp;&nbsp;Cramer implements Cramer's rule of finding the solution to a system of
&nbsp;&nbsp;&nbsp;&nbsp;linear equations Other faster solutions will be added over time

**func Determinant(m Matrix) (float64, error)**    
&nbsp;&nbsp;&nbsp;&nbsp;Determinant - normal matrix determinant The determinant and an error is
&nbsp;&nbsp;&nbsp;&nbsp;returned This is the currently the slow recursive solution (will be
&nbsp;&nbsp;&nbsp;&nbsp;amended later)

**func Equal(m1, m2 Matrix) bool**    
&nbsp;&nbsp;&nbsp;&nbsp;Equal will return whether two matrices are the same This is not too
&nbsp;&nbsp;&nbsp;&nbsp;useful seeing that one has rounding errors Will add another one to
&nbsp;&nbsp;&nbsp;&nbsp;measure equality within certain margins later

**func GetCol(m Matrix, n int) []float64**    
&nbsp;&nbsp;&nbsp;&nbsp;GetCol will extract the row n from the matrix and return it as a slice

**func GetRow(m Matrix, n int) []float64**    
&nbsp;&nbsp;&nbsp;&nbsp;GetRow will extract the row n from the matrix and return it as a slice

**func IsDiagonal(m Matrix) bool**    
&nbsp;&nbsp;&nbsp;&nbsp;IsDiagonal check whether only the diagonal has values

**func IsIdentity(m Matrix) bool**    
&nbsp;&nbsp;&nbsp;&nbsp;IsIdentity check whether the matrix is an identity matrix

**func IsLowerTriangular(m Matrix) bool**    
&nbsp;&nbsp;&nbsp;&nbsp;IsLowerTriangular check whether the left lower corner only has values

**func IsOrthogonal(m Matrix) bool**    
&nbsp;&nbsp;&nbsp;&nbsp;IsOrthogonal check whether the transpose of a matrix equals the inverse

**func IsSingleEntry(m Matrix) bool**    
&nbsp;&nbsp;&nbsp;&nbsp;IsSingleEntry check if only one cell has a value

**func IsSquare(m Matrix) bool**    
&nbsp;&nbsp;&nbsp;&nbsp;IsSquare returns whether the matrix has the same number of rows as
&nbsp;&nbsp;&nbsp;&nbsp;columns

**func IsSymmetric(m Matrix) bool**    
&nbsp;&nbsp;&nbsp;&nbsp;IsSymmetric check whether the matrix equals its transpose

**func IsUpperTriangular(m Matrix) bool**    
&nbsp;&nbsp;&nbsp;&nbsp;IsUpperTriangular check whether the right upper corner only has values

**func Print(w io.Writer, m Matrix, wdth, aft int)**    
&nbsp;&nbsp;&nbsp;&nbsp;Print will print the matrix to the w stream with wdth size for each cell
&nbsp;&nbsp;&nbsp;&nbsp;and aft decimals

**func Rows(m Matrix) int**    
&nbsp;&nbsp;&nbsp;&nbsp;Rows is an utility function to return the number of rows within the
&nbsp;&nbsp;&nbsp;&nbsp;Matrix

**func SetCol(m *Matrix, n int, val []float64)**    
&nbsp;&nbsp;&nbsp;&nbsp;SetCol will alter m by setting the n'th col to val Nothing is returned,
&nbsp;&nbsp;&nbsp;&nbsp;m is changed The number of rows of the matrix must be the same as the
&nbsp;&nbsp;&nbsp;&nbsp;length of val

**func SetRow(m *Matrix, n int, val []float64)**    
&nbsp;&nbsp;&nbsp;&nbsp;SetRow will alter m by setting the n'th row to val Nothing is returned,
&nbsp;&nbsp;&nbsp;&nbsp;m is changed The number of columns of the matrix must be the same as the
&nbsp;&nbsp;&nbsp;&nbsp;length of val

**func ToString(m Matrix, wdth, aft int) []string**    
&nbsp;&nbsp;&nbsp;&nbsp;ToString will generate a string representation of the matrix with each
&nbsp;&nbsp;&nbsp;&nbsp;cell of size wdth and aft decimals The lines are returned as a slice

**func Trace(m Matrix) float64**    
&nbsp;&nbsp;&nbsp;&nbsp;Trace returns the sum of the diagonal values

**func Add(m ...Matrix) Matrix**    
&nbsp;&nbsp;&nbsp;&nbsp;Add does a cell wise addition of the matrices returning the result If
&nbsp;&nbsp;&nbsp;&nbsp;the dimensions are not the same a nil Matrix is returned

**func AddScalar(m Matrix, n float64) Matrix**    
&nbsp;&nbsp;&nbsp;&nbsp;AddScalar will add the value n to each element within the matrix and
&nbsp;&nbsp;&nbsp;&nbsp;return the result

**func Clone(m Matrix) Matrix**    
&nbsp;&nbsp;&nbsp;&nbsp;Clone will make an exact copy of a Matrix

**func ColVectorToMatrix(v []float64) Matrix**    
&nbsp;&nbsp;&nbsp;&nbsp;ColVectorToMatrix will take the slice v an return it as a Matrix with
&nbsp;&nbsp;&nbsp;&nbsp;only one column

**func CreateMatrix(rows, cols int) Matrix**    
&nbsp;&nbsp;&nbsp;&nbsp;CreateMatrix will create and init the 2 dimensional slice

**func DivDot(m1, m2 Matrix) Matrix**    
&nbsp;&nbsp;&nbsp;&nbsp;DivDot will do a cell wise division between two matrices The rows and
&nbsp;&nbsp;&nbsp;&nbsp;columns must be the same

**func DivScalar(m Matrix, n float64) Matrix**    
&nbsp;&nbsp;&nbsp;&nbsp;DivScalar will divide each element by the value n

**func DotFunc2(m1, m2 Matrix, f func(float64, float64) float64) Matrix**    
&nbsp;&nbsp;&nbsp;&nbsp;DotFunc2 will take cells from the two matrices and apply f to it
&nbsp;&nbsp;&nbsp;&nbsp;generating a new matrix that is returned m1 and m2 must be the same
&nbsp;&nbsp;&nbsp;&nbsp;dimensions

**func DotFuncN(f func(v ...float64) float64, m ...Matrix) Matrix**    
&nbsp;&nbsp;&nbsp;&nbsp;DotFuncN will take cells from all the matrices and apply f to it
&nbsp;&nbsp;&nbsp;&nbsp;generating a new matrix that is returned It is necessary to make sure
&nbsp;&nbsp;&nbsp;&nbsp;that the number of variables that f expect is the same as the number of
&nbsp;&nbsp;&nbsp;&nbsp;matrices provided all matrices must be the same dimensions This is
&nbsp;&nbsp;&nbsp;&nbsp;similar to the map function used by various lisp and other languages

**func Exclude(m Matrix, row, col int) Matrix**    
&nbsp;&nbsp;&nbsp;&nbsp;Exclude will return a new matrix with the specific row and column
&nbsp;&nbsp;&nbsp;&nbsp;excluded from it

**func IdentityMatrix(n int) Matrix**    
&nbsp;&nbsp;&nbsp;&nbsp;IndentiyMatrix will generate an identity matrix of dimension nxn

**func Inverse(m Matrix) (Matrix, error)**    
&nbsp;&nbsp;&nbsp;&nbsp;Inverse will calculate the inverse of a matrix if it is square and the
&nbsp;&nbsp;&nbsp;&nbsp;determinant is not zero

**func InvertHorizontal(m Matrix) Matrix**    
&nbsp;&nbsp;&nbsp;&nbsp;InvertHorizontal will invert each row of m

**func InvertVertical(m Matrix) Matrix**    
&nbsp;&nbsp;&nbsp;&nbsp;InvertVertical will invert each column of m

**func Mult(m ...Matrix) Matrix**    
&nbsp;&nbsp;&nbsp;&nbsp;Mult will multiply the provided matrices with each other (using Mult2)

**func Mult2(m1 Matrix, m2 Matrix) Matrix**    
&nbsp;&nbsp;&nbsp;&nbsp;Mult2 will multiply two matrices This is the standard slow method, the
&nbsp;&nbsp;&nbsp;&nbsp;faster alternatives will be added later Note that the columns of m1 must
&nbsp;&nbsp;&nbsp;&nbsp;be the same as the rows of m2

**func MultDot(m1, m2 Matrix) Matrix **    
&nbsp;&nbsp;&nbsp;&nbsp;MultDot will do a cell wise multiplication between two matrices The rows
&nbsp;&nbsp;&nbsp;&nbsp;and columns must be the same

**func MultScalar(m1 Matrix, val float64) Matrix**    
&nbsp;&nbsp;&nbsp;&nbsp;MultScalar multiplies the value val to each cell of m1 and return the
&nbsp;&nbsp;&nbsp;&nbsp;result

**func Pow(m Matrix, n int) Matrix**    
&nbsp;&nbsp;&nbsp;&nbsp;Pow will return the integer power of a matrix if it is square

**func PowDot(m1, m2 Matrix) Matrix**    
&nbsp;&nbsp;&nbsp;&nbsp;PowDot will calc the exponent of the cell in m1 to the cell in m2
&nbsp;&nbsp;&nbsp;&nbsp;returning the result The rows and columns must be the same

**func PowScalar(m Matrix, n float64) Matrix**    
&nbsp;&nbsp;&nbsp;&nbsp;PowScalar will take the exponent of each cell to the value n returning
&nbsp;&nbsp;&nbsp;&nbsp;the new matrix

**func RowVectorToMatrix(v []float64) Matrix**    
&nbsp;&nbsp;&nbsp;&nbsp;RowVectorToMatrix will take the slice v and return it as a Matrix with
&nbsp;&nbsp;&nbsp;&nbsp;only one row

**func ScalarFunc(m Matrix, f func(float64) float64) Matrix**    
&nbsp;&nbsp;&nbsp;&nbsp;Mult**func will apply a **func to each cell return the new matrix The func
&nbsp;&nbsp;&nbsp;&nbsp;in this instance has only one value

**func Sub(m ...Matrix) Matrix**    
&nbsp;&nbsp;&nbsp;&nbsp;Add does a cell wise subtraction of the matrices returning the result If
&nbsp;&nbsp;&nbsp;&nbsp;the dimensions are not the same a nil Matrix is returned

**func SubMatrix(m Matrix, strow, stcol, rows, cols int) Matrix**    
&nbsp;&nbsp;&nbsp;&nbsp;SubMatrix will extract a piece of a matrix strow - start row stcol -
&nbsp;&nbsp;&nbsp;&nbsp;start col rows - number of rows to take cols - number of cols to take

**func SubScalar(m Matrix, n float64) Matrix**    
&nbsp;&nbsp;&nbsp;&nbsp;SubScalar will subtract the value n from each cell returning the result

**func SwapCols(m Matrix, col1, col2 int) Matrix**    
&nbsp;&nbsp;&nbsp;&nbsp;SwapCols will swap columns col1 and col2 and return the result

**func SwapRows(m Matrix, row1, row2 int) Matrix**    
&nbsp;&nbsp;&nbsp;&nbsp;SwapRows will swap rows row1 and row2 and return the result

**func Transpose(m Matrix) Matrix**    
&nbsp;&nbsp;&nbsp;&nbsp;Transpose will do a normal matrix transpose and return it

