// hduplooy/gomatrix
// Author: Hannes du Plooy
// Basic matrix handling library (Work in Progress)
//
// Revision Date: 19 Sep 2019
package gomatrix

import (
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
)

// A Matrix is just a 2 dimensional slice of type float64
type Matrix [][]float64

// CreateMatrix will create and init the 2 dimensional slice
func CreateMatrix(rows, cols int) Matrix {
	m := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		m[i] = make([]float64, cols)
	}
	return m
}

// Clone will make an exact copy of a Matrix
func Clone(m Matrix) Matrix {
	rows := Rows(m)
	m2 := CreateMatrix(rows, Cols(m))
	for i := 0; i < rows; i++ {
		copy(m2[i], m[i])
	}
	return m2
}

// Rows is an utility function to return the number of rows within the Matrix
func Rows(m Matrix) int {
	if m == nil {
		return 0
	}
	return len(m)
}

// Cols is an utility function to return the number of columns within the Matrix
func Cols(m Matrix) int {
	if m == nil {
		return 0
	}
	if len(m) > 0 {
		return len(m[0])
	}
	return 0
}

// AddScalar will add the value n to each element within the matrix and return the result
func AddScalar(m Matrix, n float64) Matrix {
	if m == nil {
		return nil
	}
	m2 := Clone(m)
	for i := 0; i < Rows(m); i++ {
		for j := 0; j < Cols(m); j++ {
			m2[i][j] += n
		}
	}
	return m2
}

// Add does a cell wise addition of the matrices returning the result
// If the dimensions are not the same a nil Matrix is returned
func Add(m ...Matrix) Matrix {
	if len(m) == 0 {
		return nil
	}
	ans := Clone(m[0])
	rows := Rows(m[0])
	cols := Cols(m[0])
	for _, val := range m[1:] {
		if rows != Rows(val) || cols != Cols(val) {
			return nil
		}
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				ans[i][j] += val[i][j]
			}
		}
	}
	return ans
}

// SubScalar will subtract the value n from each cell returning the result
func SubScalar(m Matrix, n float64) Matrix {
	if m == nil {
		return nil
	}
	m2 := Clone(m)
	for i := 0; i < Rows(m); i++ {
		for j := 0; j < Cols(m); j++ {
			m2[i][j] -= n
		}
	}
	return m2
}

// Add does a cell wise subtraction of the matrices returning the result
// If the dimensions are not the same a nil Matrix is returned
func Sub(m ...Matrix) Matrix {
	if len(m) == 0 {
		return nil
	}
	ans := Clone(m[0])
	rows := Rows(m[0])
	cols := Cols(m[0])
	for _, val := range m[1:] {
		if rows != Rows(val) || cols != Cols(val) {
			return nil
		}
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				ans[i][j] -= val[i][j]
			}
		}
	}
	return ans
}

// MultScalar multiplies the value val to each cell of m1 and return the result
func MultScalar(m1 Matrix, val float64) Matrix {
	rows := Rows(m1)
	cols := Cols(m1)
	if m1 == nil || rows == 0 || cols == 0 {
		return nil
	}
	m2 := CreateMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m2[i][j] = m1[i][j] * val
		}
	}
	return m2
}

// Mult2 will multiply two matrices
// This is the standard slow method, the faster alternatives will be added later
// Note that the columns of m1 must be the same as the rows of m2
func Mult2(m1 Matrix, m2 Matrix) Matrix {
	if Cols(m1) != Rows(m2) {
		return nil
	}
	m3 := CreateMatrix(Rows(m1), Cols(m2))
	for i := 0; i < Rows(m1); i++ {
		for j := 0; j < Cols(m2); j++ {
			tmp := float64(0)
			for k := 0; k < Cols(m1); k++ {
				tmp += m1[i][k] * m2[k][j]
			}
			m3[i][j] = tmp
		}
	}
	return m3
}

// Mult will multiply the provided matrices with each other (using Mult2)
func Mult(m ...Matrix) Matrix {
	if len(m) < 2 {
		return nil
	}
	ans := Mult2(m[0], m[1])
	if ans == nil {
		return nil
	}
	for _, val := range m[2:] {
		ans = Mult2(ans, val)
		if ans == nil {
			return nil
		}
	}
	return ans
}

// DivScalar will divide each element by the value n
func DivScalar(m Matrix, n float64) Matrix {
	if m == nil {
		return nil
	}
	m2 := Clone(m)
	for i := 0; i < Rows(m); i++ {
		for j := 0; j < Cols(m); j++ {
			m2[i][j] -= n
		}
	}
	return m2
}

// PowScalar will take the exponent of each cell to the value n returning the new matrix
func PowScalar(m Matrix, n float64) Matrix {
	if m == nil {
		return nil
	}
	m2 := Clone(m)
	for i := 0; i < Rows(m); i++ {
		for j := 0; j < Cols(m); j++ {
			m2[i][j] = math.Pow(m2[i][j], n)
		}
	}
	return m2
}

// MultDot will do a cell wise multiplication between two matrices
// The rows and columns must be the same
func MultDot(m1, m2 Matrix) Matrix {
	rows := Rows(m1)
	cols := Cols(m1)
	if m1 == nil || m2 == nil || rows != Rows(m2) || cols != Cols(m2) {
		return nil
	}
	m3 := CreateMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m3[i][j] = m1[i][j] * m2[i][j]
		}
	}
	return m3
}

// DivDot will do a cell wise division between two matrices
// The rows and columns must be the same
func DivDot(m1, m2 Matrix) Matrix {
	rows := Rows(m1)
	cols := Cols(m1)
	if rows != Rows(m2) || cols != Cols(m2) {
		return nil
	}
	m3 := CreateMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m3[i][j] = m1[i][j] / m2[i][j]
		}
	}
	return m3
}

// PowDot will calc the exponent of the cell in m1 to the cell in m2 returning the result
// The rows and columns must be the same
func PowDot(m1, m2 Matrix) Matrix {
	rows := Rows(m1)
	cols := Cols(m1)
	if rows != Rows(m2) || cols != Cols(m2) {
		return nil
	}
	m3 := CreateMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m3[i][j] = math.Pow(m1[i][j], m2[i][j])
		}
	}
	return m3
}

// MultFunc will apply a func to each cell return the new matrix
// The func in this instance has only one value
func ScalarFunc(m Matrix, f func(float64) float64) Matrix {
	if m == nil {
		return nil
	}
	m2 := Clone(m)
	for i := 0; i < Rows(m); i++ {
		for j := 0; j < Cols(m); j++ {
			m2[i][j] = f(m2[i][j])
		}
	}
	return m2
}

// DotFunc2 will take cells from the two matrices and apply f to it generating a new matrix
// that is returned
// m1 and m2 must be the same dimensions
func DotFunc2(m1, m2 Matrix, f func(float64, float64) float64) Matrix {
	if m1 == nil || m2 == nil || Rows(m1) != Rows(m2) || Cols(m1) != Cols(m2) {
		return nil
	}
	m3 := CreateMatrix(Rows(m1), Cols(m2))
	for i := 0; i < Rows(m3); i++ {
		for j := 0; j < Cols(m3); j++ {
			m3[i][j] = f(m1[i][j], m2[i][j])
		}
	}
	return m3
}

// DotFuncN will take cells from all the matrices and apply f to it generating a new matrix
// that is returned
// It is necessary to make sure that the number of variables that f expect is the same as the number
// of matrices provided
// all matrices must be the same dimensions
// This is similar to the map function used by various lisp and other languages
func DotFuncN(f func(v ...float64) float64, m ...Matrix) Matrix {
	if len(m) < 1 || m[0] == nil {
		return nil
	}
	rows := Rows(m[0])
	cols := Cols(m[0])
	for _, val := range m[1:] {
		if rows != Rows(val) || cols != Cols(val) {
			return nil
		}
	}
	m2 := CreateMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			vals := make([]float64, len(m))
			for k := 0; k < len(m); k++ {
				vals[k] = m[k][i][j]
			}
			m2[i][j] = f(vals...)
		}
	}
	return m2
}

// Transpose will do a normal matrix transpose and return it
func Transpose(m Matrix) Matrix {
	if m == nil || Cols(m) == 0 || Rows(m) == 0 {
		return nil
	}
	m1 := CreateMatrix(Cols(m), Rows(m))
	for i := 0; i < Rows(m); i++ {
		for j := 0; j < Cols(m); j++ {
			m1[i][j] = m[j][i]
		}
	}
	return m1
}

// RowVectorToMatrix will take the slice v and return it as a Matrix with only one row
func RowVectorToMatrix(v []float64) Matrix {
	if v == nil {
		return nil
	}
	m := CreateMatrix(1, len(v))
	copy(m[0], v)
	return m
}

// ColVectorToMatrix will take the slice v an return it as a Matrix with only one column
func ColVectorToMatrix(v []float64) Matrix {
	if v == nil {
		return nil
	}
	m := CreateMatrix(len(v), 1)
	for i := 0; i < len(v); i++ {
		m[i][0] = v[i]
	}
	return m
}

// CloneVector just makes a copy of a slice
func CloneVector(val []float64) []float64 {
	tmp := make([]float64, len(val))
	copy(tmp, val)
	return tmp
}

// GetRow will extract the row n from the matrix and return it as a slice
func GetRow(m Matrix, n int) []float64 {
	if m == nil {
		return nil
	}
	if n >= 0 && n < len(m) {
		return CloneVector(m[n])
	}
	return nil
}

// GetCol will extract the row n from the matrix and return it as a slice
func GetCol(m Matrix, n int) []float64 {
	if m == nil {
		return nil
	}
	if n >= 0 && n < Cols(m) {
		tmp := make([]float64, Rows(m))
		for i := 0; i < Rows(m); i++ {
			tmp[i] = m[i][n]
		}
		return tmp
	}
	return nil
}

// SetRow will alter m by setting the n'th row to val
// Nothing is returned, m is changed
// The number of columns of the matrix must be the same as the length of val
func SetRow(m *Matrix, n int, val []float64) {
	if m == nil || val == nil || Cols(*m) != len(val) {
		return
	}
	if n >= 0 && n < Rows(*m) {
		copy((*m)[n], val)
	}
}

// SetCol will alter m by setting the n'th col to val
// Nothing is returned, m is changed
// The number of rows of the matrix must be the same as the length of val
func SetCol(m *Matrix, n int, val []float64) {
	if m == nil || val == nil || Rows(*m) != len(val) {
		return
	}
	if n >= 0 && n < Cols(*m) {
		tmp := len(val)
		for i := 0; i < tmp; i++ {
			(*m)[i][n] = val[i]
		}
	}
}

// IsSquare returns whether the matrix has the same number of rows as columns
func IsSquare(m Matrix) bool {
	return Rows(m) == Cols(m)
}

// IndentiyMatrix will generate an identity matrix of dimension nxn
func IdentityMatrix(n int) Matrix {
	m := CreateMatrix(n, n)
	for i := 0; i < n; i++ {
		m[i][i] = 1.0
	}
	return m
}

// Equal will return whether two matrices are the same
// This is not too useful seeing that one has rounding errors
// Will add another one to measure equality within certain margins later
func Equal(m1, m2 Matrix) bool {
	if Rows(m1) != Rows(m2) || Cols(m1) != Cols(m2) {
		return false
	}
	for i := 0; i < Rows(m1); i++ {
		for j := 0; j < Cols(m1); j++ {
			if m1[i][j] != m2[i][j] {
				return false
			}
		}
	}
	return true
}

// IsIdentity check whether the matrix is an identity matrix
func IsIdentity(m Matrix) bool {
	if m == nil {
		return false
	}
	for i := 0; i < Rows(m); i++ {
		for j := 0; j < Cols(m); j++ {
			if i == j {
				if m[i][j] != 1 {
					return false
				}
			} else {
				if m[i][j] != 0 {
					return false
				}
			}
		}
	}
	return true
}

// IsDiagonal check whether only the diagonal has values
func IsDiagonal(m Matrix) bool {
	if m == nil {
		return false
	}
	tmp := false
	for i := 0; i < Rows(m); i++ {
		for j := 0; j < Cols(m); j++ {
			if i == j {
				if m[i][j] != 0 {
					tmp = true
				}
			} else {
				if m[i][j] != 0 {
					return false
				}
			}
		}
	}
	return tmp
}

// IsUpperTriangular check whether the right upper corner only has values
func IsUpperTriangular(m Matrix) bool {
	if m == nil {
		return false
	}
	tmp := false
	for i := 0; i < Rows(m); i++ {
		for j := 0; j < Rows(m); j++ {
			if j <= i {
				if m[i][j] != 0 {
					tmp = true
				}
			} else {
				if m[i][j] != 0 {
					return false
				}
			}
		}
	}
	return tmp
}

// IsLowerTriangular check whether the left lower corner only has values
func IsLowerTriangular(m Matrix) bool {
	if m == nil {
		return false
	}
	tmp := false
	for i := 0; i < Rows(m); i++ {
		for j := 0; j < Rows(m); j++ {
			if j >= i {
				if m[i][j] != 0 {
					tmp = true
				}
			} else {
				if m[i][j] != 0 {
					return false
				}
			}
		}
	}
	return tmp
}

// IsSymmetric check whether the matrix equals its transpose
func IsSymmetric(m Matrix) bool {
	if m == nil {
		return false
	}
	for i := 0; i < Rows(m); i++ {
		for j := 0; j < i; j++ {
			if m[i][j] != m[j][i] {
				return false
			}
		}
	}
	return true
}

// IsOrthogonal check whether the transpose of a matrix equals the inverse
func IsOrthogonal(m Matrix) bool {
	if m == nil {
		return false
	}
	tmp, err := Inverse(m)
	if err != nil {
		return false
	}
	return Equal(Transpose(m), tmp)
}

// IsSingleEntry check if only one cell has a value
func IsSingleEntry(m Matrix) bool {
	if m == nil {
		return false
	}
	cnt := 0
	for i := 0; i < Rows(m); i++ {
		for j := 0; j < Cols(m); j++ {
			if m[i][j] != 0 {
				cnt++
			}
			if cnt > 0 {
				break
			}
		}
		if cnt > 0 {
			break
		}
	}
	return cnt == 1
}

// Trace returns the sum of the diagonal values
func Trace(m Matrix) float64 {
	if IsSquare(m) {
		ans := 0.0
		for i := 0; i < len(m); i++ {
			ans += m[i][i]
		}
		return ans
	}
	return 0
}

// Exclude will return a new matrix with the specific row and column excluded from it
func Exclude(m Matrix, row, col int) Matrix {
	if row < 0 || row >= Rows(m) || col < 0 || col >= Cols(m) {
		return nil
	}
	m2 := CreateMatrix(Rows(m)-1, Cols(m)-1)
	for i := 0; i < Rows(m); i++ {
		if i != row {
			for j := 0; j < Cols(m); j++ {
				if j != col {
					r := i
					if r > row {
						r--
					}
					c := j
					if c > col {
						c--
					}
					m2[r][c] = m[i][j]
				}
			}
		}
	}
	return m2
}

// Determinant - normal matrix determinant
// The determinant and an error is returned
// This is the currently the slow recursive solution (will be amended later)
func Determinant(m Matrix) (float64, error) {
	if !IsSquare(m) {
		return 0, errors.New("Determinant limited to square matrices")
	}
	if Rows(m) == 1 {
		return m[0][0], nil
	}
	if Rows(m) == 2 {
		return m[0][0]*m[1][1] - m[0][1]*m[1][0], nil
	}
	ans := 0.0
	sgn := 1.0
	for i := 0; i < Cols(m); i++ {
		det, err := Determinant(Exclude(m, 0, i))
		if err != nil {
			return 0, err
		}
		ans += sgn * m[0][i] * det
		sgn = -sgn
	}
	return ans, nil
}

// Inverse will calculate the inverse of a matrix if it is square and the determinant is not zero
func Inverse(m Matrix) (Matrix, error) {
	if !IsSquare(m) {
		return nil, errors.New("Matrix must be square")
	}
	mdet, err := Determinant(m)
	if err != nil {
		return nil, err
	}
	if mdet == 0 {
		return nil, errors.New("The determinant of an invertible matrix is not zero")
	}
	m2 := CreateMatrix(Rows(m), Cols(m))
	for i := 0; i < Rows(m); i++ {
		for j := 0; j < Cols(m); j++ {
			sgn := 1.0
			if (i+j)%2 == 1 {
				sgn = -sgn
			}
			det, err := Determinant(Exclude(m, i, j))
			if err != nil {
				return nil, err
			}
			m2[j][i] = sgn * det / mdet
		}
	}
	return m2, nil
}

// Pow will return the integer power of a matrix if it is square
func Pow(m Matrix, n int) Matrix {
	if m == nil || n < 0 || !IsSquare(m) {
		return nil
	}
	if n == 0 {
		return IdentityMatrix(Rows(m))
	}
	if n == 1 {
		return Clone(m)
	}
	if n%2 == 0 {
		tmp := Pow(m, n/2)
		return Mult(tmp, tmp)
	}
	tmp := Pow(m, (n-1)/2)
	return Mult(tmp, Mult(tmp, m))
}

// SubMatrix will extract a piece of a matrix
// strow - start row
// stcol - start col
// rows - number of rows to take
// cols - number of cols to take
func SubMatrix(m Matrix, strow, stcol, rows, cols int) Matrix {
	if m == nil || strow < 0 || stcol < 0 || rows < 0 || cols < 0 || strow+rows > Rows(m) || stcol+cols > Cols(m) {
		return nil
	}
	m2 := CreateMatrix(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m2[i][j] = m[strow+i][stcol+j]
		}
	}
	return m2
}

// SwapRows will swap rows row1 and row2 and return the result
func SwapRows(m Matrix, row1, row2 int) Matrix {
	if m == nil || row1 < 0 || row2 < 0 || row1 >= Rows(m) || row2 >= Rows(m) {
		return nil
	}
	m2 := Clone(m)
	for i := 0; i < Cols(m2); i++ {
		m2[row1][i], m2[row2][i] = m2[row2][i], m2[row1][i]
	}
	return m2
}

// SwapCols will swap columns col1 and col2 and return the result
func SwapCols(m Matrix, col1, col2 int) Matrix {
	if m == nil || col1 < 0 || col2 < 0 || col1 >= Cols(m) || col2 >= Cols(m) {
		return nil
	}
	m2 := Clone(m)
	for i := 0; i < Rows(m2); i++ {
		m2[i][col1], m2[i][col2] = m2[i][col2], m2[i][col1]
	}
	return m2
}

// InvertHorizontal will invert each row of m
func InvertHorizontal(m Matrix) Matrix {
	if m == nil {
		return nil
	}
	m2 := Clone(m)
	for k := 0; k < Rows(m); k++ {
		for i, j := 0, Cols(m2)-1; i < j; i, j = i+1, j-1 {
			m2[k][i], m2[k][i] = m2[k][j], m2[k][i]
		}
	}
	return m2
}

// InvertVertical will invert each column of m
func InvertVertical(m Matrix) Matrix {
	if m == nil {
		return nil
	}
	m2 := Clone(m)
	for i, j := 0, Rows(m2)-1; i < j; i, j = i+1, j-1 {
		m2[i], m2[j] = m2[j], m2[i]
	}
	return m2
}

// ToString will generate a string representation of the matrix with each cell of size wdth and aft decimals
// The lines are returned as a slice
func ToString(m Matrix, wdth, aft int) []string {
	ans := make([]string, 0, 10)
	for i := 0; i < Rows(m); i++ {
		tmp := "|"
		for j := 0; j < Cols(m); j++ {
			if j > 0 {
				tmp += " "
			}
			tmp += fmt.Sprintf("%*.*f", wdth, aft, m[i][j])
		}
		tmp += "|"
		ans = append(ans, tmp)
	}
	return ans
}

// Print will print the matrix to the w stream with wdth size for each cell and aft decimals
func Print(w io.Writer, m Matrix, wdth, aft int) {
	fmt.Fprint(w, strings.Join(ToString(m, wdth, aft), "\n"))
}

// Cramer implements Cramer's rule of finding the solution to a system of linear equations
// Other faster solutions will be added over time
func Cramer(m Matrix, r []float64) ([]float64, error) {
	if m == nil {
		return nil, errors.New("Invalid matrix for Cramer")
	}
	if !IsSquare(m) {
		return nil, errors.New("Matrix for Cramer is not Square")
	}
	if Rows(m) != len(r) {
		return nil, errors.New("The rows/cols of the matrix does not equal the number of elements in the vector for Cramer")
	}
	ans := make([]float64, len(r))
	tmp, err := Determinant(m)
	if err != nil {
		return nil, err
	}
	for i := 0; i < len(r); i++ {
		m2 := Clone(m)
		SetCol(&m2, i, r)
		tmp2, err := Determinant(m2)
		if err != nil {
			return nil, err
		}
		ans[i] = tmp2 / tmp
	}
	return ans, nil
}
