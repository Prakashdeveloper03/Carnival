package storage

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
