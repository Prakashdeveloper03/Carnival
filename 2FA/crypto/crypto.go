package crypto

import (
	"crypto/rand"
	"fmt"
	"io"

	"golang.org/x/crypto/scrypt"
)

const (
	KeySize  = 32
	SaltSize = 16
)

func GenerateSalt() ([]byte, error) {
	salt := make([]byte, SaltSize)
	if _, err := io.ReadFull(rand.Reader, salt); err != nil {
		return nil, fmt.Errorf("failed to generate salt: %w", err)
	}
	return salt, nil
}

func GenerateKey(password, salt []byte) ([]byte, error) {

	return scrypt.Key(password, salt, 32768, 8, 1, KeySize)
}
