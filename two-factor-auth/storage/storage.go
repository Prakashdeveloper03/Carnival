// Package storage manages reading from and writing to the encrypted secrets file.
package storage

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// StoredAccount represents the structure of an account as it is saved on disk.
// The secret is stored in an encrypted format.
type StoredAccount struct {
	Issuer          string `json:"issuer"`
	AccountName     string `json:"account_name"`
	EncryptedSecret []byte `json:"encrypted_secret"`
}

// StoredData is the top-level structure for the data saved to the JSON file.
type StoredData struct {
	Salt     []byte          `json:"salt"`
	Accounts []StoredAccount `json:"accounts"`
}

// Store handles the file path and I/O operations for the application's data.
type Store struct {
	// FilePath is the full, absolute path to the secrets file.
	FilePath string
}

// NewStore creates a new Store instance, determining the correct file path.
func NewStore() (*Store, error) {
	configDir, err := os.UserConfigDir()
	if err != nil {
		return nil, fmt.Errorf("cannot find user config directory: %w", err)
	}

	appDir := filepath.Join(configDir, "go-2fa-cli")
	if err := os.MkdirAll(appDir, 0700); err != nil {
		return nil, fmt.Errorf("cannot create app config directory: %w", err)
	}

	return &Store{
		FilePath: filepath.Join(appDir, "secrets.json"),
	}, nil
}

// Path returns the file path of the store.
func (s *Store) Path() string {
	return s.FilePath
}

// Exists checks if the store's file already exists.
func (s *Store) Exists() bool {
	_, err := os.Stat(s.FilePath)
	return !os.IsNotExist(err)
}

// Load reads and unmarshals the StoredData from the file.
func (s *Store) Load() (*StoredData, error) {
	fileBytes, err := os.ReadFile(s.FilePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read store file: %w", err)
	}

	var data StoredData
	if err := json.Unmarshal(fileBytes, &data); err != nil {
		return nil, fmt.Errorf("failed to parse store file: %w", err)
	}
	return &data, nil
}

// Save marshals the provided StoredData and writes it to the file.
func (s *Store) Save(data *StoredData) error {
	fileBytes, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal data: %w", err)
	}

	// Write with permissions that only the current user can read/write.
	if err := os.WriteFile(s.FilePath, fileBytes, 0600); err != nil {
		return fmt.Errorf("failed to write store file: %w", err)
	}
	return nil
}
