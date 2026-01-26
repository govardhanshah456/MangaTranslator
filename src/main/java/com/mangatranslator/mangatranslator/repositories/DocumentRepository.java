package com.mangatranslator.mangatranslator.repositories;

import com.mangatranslator.mangatranslator.models.Document;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface DocumentRepository extends JpaRepository<Document, String> {
}
