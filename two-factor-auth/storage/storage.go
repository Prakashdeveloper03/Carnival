package storage

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

type StoredAccount struct {
	Issuer          string `json:"issuer"`
	AccountName     string `json:"account_name"`
	EncryptedSecret []byte `json:"encrypted_secret"`
}

type StoredData struct {
	Salt     []byte          `json:"salt"`
	Accounts []StoredAccount `json:"accounts"`
}

type Store struct {
	filePath string
}

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
		filePath: filepath.Join(appDir, "secrets.json"),
	}, nil
}

func (s *Store) Path() string {
	return s.filePath
}

func (s *Store) Exists() bool {
	_, err := os.Stat(s.filePath)
	return !os.IsNotExist(err)
}

func (s *Store) Load() (*StoredData, error) {
	fileBytes, err := os.ReadFile(s.filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read store file: %w", err)
	}

	var data StoredData
	if err := json.Unmarshal(fileBytes, &data); err != nil {
		return nil, fmt.Errorf("failed to parse store file: %w", err)
	}
	return &data, nil
}

func (s *Store) Save(data *StoredData) error {
	fileBytes, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal data: %w", err)
	}

	if err := os.WriteFile(s.filePath, fileBytes, 0600); err != nil {
		return fmt.Errorf("failed to write store file: %w", err)
	}
	return nil
}
